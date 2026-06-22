#include "../op/utils.h"
#include "common/ast_traverser.h"
#include "tvm/arith/pattern.h"
#include "tvm/ffi/reflection/registry.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/analysis.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"

#include <highs/Highs.h>
#include <highs/lp_data/HConst.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

enum class Role : uint8_t { kConsumer, kProducer, kBoth, kUndefined };

struct AccessInfo {
  BufferRegion region;
  bool is_write{false};
  int iter_offset{0};

  const Buffer& buffer() const { return region->buffer; }
};

enum class DeviceType {
  ODMA,
  TensorCore,
  VectorCore,
  Unspecified,
};

enum class IlpResourceType : int {
  kTensorCore = 0,
  kVectorCore = 1,
  kODMA0 = 2,
  kODMA1 = 3,
  kWsramIn = 6,
  kWsramOut = 7,
  kAsramIn = 8,
  kAsramOut = 9,
  kRsram = 10,
};

struct CommandSpec {
  int latency{0};
  std::vector<int> resources;
  std::string name;
};

struct FlowSpec {
  bool resident{false};
  int prod{-1};
  int cons{-1};
  int mem{0};
  int fixed_bank{-1};
  int fp{1};
  int initial_time{0};
  int w_off{0};
  int w_dur{0};
  int r_off{0};
  int r_dur{0};
  int write_resource{-1};
  int read_resource{-1};
};

struct Problem {
  int N{0};
  int Tmax{0};
  std::vector<int> R;
  std::unordered_map<int, int> cap;
  std::vector<CommandSpec> P;
  std::vector<std::pair<int, int>> dep_edges;
  std::unordered_map<long long, int> delta;
  std::vector<FlowSpec> flows;
};

struct ModelVars {
  HighsInt col_T{-1};
  std::vector<HighsInt> col_t;
  std::vector<HighsInt> col_y;
  std::vector<HighsInt> col_m;
  std::vector<std::vector<HighsInt>> col_x;
  std::vector<std::vector<HighsInt>> col_a;
  std::vector<int> internal_flow_ids;
  std::vector<int> resident_flow_ids;
  std::vector<std::array<HighsInt, 2>> col_z;
  std::vector<std::vector<HighsInt>> col_q;
  std::vector<std::vector<HighsInt>> col_alpha;
  std::vector<std::vector<HighsInt>> col_beta;
  std::vector<std::vector<std::vector<HighsInt>>> col_h;
  std::vector<std::vector<std::vector<HighsInt>>> col_w;
  std::vector<std::vector<std::vector<HighsInt>>> col_r;
};

struct SolveResult {
  bool ok{false};
  int II{0};
  int makespan{0};
  std::vector<int> t;
  std::vector<int> m;
  std::vector<int> y;
};

namespace {

// File-local helper to avoid colliding with the similarly named function in
// sunmmio_pipeline_planning.cc during final shared-library link.
bool PipelineRegionIntersect(const Region& region1, const Region& region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); ++i) {
    const Range& dim1 = region1[i];
    const Range& dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

const double kInf = kHighsInf;

long long EdgeKey(int i, int j) {
  return (static_cast<long long>(i) << 32) | static_cast<unsigned int>(j);
}

int CeilDiv(int a, int b) {
  ICHECK_GT(b, 0);
  return (a + b - 1) / b;
}

int GetEnvInt(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) {
    return default_value;
  }
  return std::atoi(raw);
}

std::string GetEnvString(const char* name) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) {
    return "";
  }
  return raw;
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(c);
        break;
    }
  }
  return escaped;
}

void WriteJsonString(std::ostream& os, const std::string& value) {
  os << "\"" << JsonEscape(value) << "\"";
}

void WriteProblemJson(const Problem& prob, const std::string& path) {
  std::ofstream out(path);
  ICHECK(out.is_open()) << "Failed to open ILP problem json path: " << path;
  out << std::boolalpha;
  out << "{\n";
  out << "  \"N\": " << prob.N << ",\n";
  out << "  \"Tmax\": " << prob.Tmax << ",\n";

  out << "  \"R\": [";
  for (size_t i = 0; i < prob.R.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << prob.R[i];
  }
  out << "],\n";

  std::map<int, int> sorted_cap(prob.cap.begin(), prob.cap.end());
  out << "  \"cap\": {";
  bool first_cap = true;
  for (const auto& kv : sorted_cap) {
    if (!first_cap) {
      out << ", ";
    }
    first_cap = false;
    WriteJsonString(out, std::to_string(kv.first));
    out << ": " << kv.second;
  }
  out << "},\n";

  out << "  \"commands\": {\n";
  for (int i = 0; i < prob.N; ++i) {
    out << "    ";
    WriteJsonString(out, std::to_string(i));
    out << ": {\"latency\": " << prob.P[i].latency << ", \"resources\": [";
    for (size_t j = 0; j < prob.P[i].resources.size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << prob.P[i].resources[j];
    }
    out << "], \"name\": ";
    WriteJsonString(out, prob.P[i].name);
    out << "}";
    out << (i + 1 == prob.N ? "\n" : ",\n");
  }
  out << "  },\n";

  out << "  \"P\": {\n";
  for (int i = 0; i < prob.N; ++i) {
    out << "    ";
    WriteJsonString(out, std::to_string(i));
    out << ": {\"latency\": " << prob.P[i].latency << ", \"resources\": [";
    for (size_t j = 0; j < prob.P[i].resources.size(); ++j) {
      if (j != 0) {
        out << ", ";
      }
      out << prob.P[i].resources[j];
    }
    out << "]}";
    out << (i + 1 == prob.N ? "\n" : ",\n");
  }
  out << "  },\n";

  out << "  \"dep_edges\": [";
  for (size_t i = 0; i < prob.dep_edges.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << "[" << prob.dep_edges[i].first << ", " << prob.dep_edges[i].second << "]";
  }
  out << "],\n";

  std::map<std::pair<int, int>, int> sorted_delta;
  for (const auto& edge : prob.dep_edges) {
    auto it = prob.delta.find(EdgeKey(edge.first, edge.second));
    if (it != prob.delta.end()) {
      sorted_delta[edge] = it->second;
    }
  }
  out << "  \"delta\": {";
  bool first_delta = true;
  for (const auto& kv : sorted_delta) {
    if (!first_delta) {
      out << ", ";
    }
    first_delta = false;
    WriteJsonString(out, std::to_string(kv.first.first) + "," +
                             std::to_string(kv.first.second));
    out << ": " << kv.second;
  }
  out << "},\n";

  std::vector<FlowSpec> sorted_flows = prob.flows;
  std::sort(sorted_flows.begin(), sorted_flows.end(),
            [](const FlowSpec& a, const FlowSpec& b) {
              return std::tie(a.resident, a.prod, a.cons, a.mem, a.fixed_bank, a.fp,
                              a.initial_time, a.w_off, a.w_dur, a.r_off, a.r_dur,
                              a.write_resource, a.read_resource) <
                     std::tie(b.resident, b.prod, b.cons, b.mem, b.fixed_bank, b.fp,
                              b.initial_time, b.w_off, b.w_dur, b.r_off, b.r_dur,
                              b.write_resource, b.read_resource);
            });
  out << "  \"flows\": [";
  for (size_t i = 0; i < sorted_flows.size(); ++i) {
    const FlowSpec& flow = sorted_flows[i];
    if (i != 0) {
      out << ", ";
    }
    out << "{";
    out << "\"kind\": ";
    WriteJsonString(out, flow.resident ? "resident" : "internal");
    out << ", \"prod\": " << flow.prod;
    out << ", \"cons\": " << flow.cons;
    out << ", \"mem\": " << flow.mem;
    out << ", \"fixed_bank\": " << flow.fixed_bank;
    out << ", \"fp\": " << flow.fp;
    out << ", \"initial_time\": " << flow.initial_time;
    out << ", \"w_off\": " << flow.w_off;
    out << ", \"w_dur\": " << flow.w_dur;
    out << ", \"r_off\": " << flow.r_off;
    out << ", \"r_dur\": " << flow.r_dur;
    out << ", \"write_resource\": " << flow.write_resource;
    out << ", \"read_resource\": " << flow.read_resource;
    out << "}";
  }
  out << "]\n";
  out << "}\n";
}

void MaybeExportProblemJson(const Problem& prob, bool debug) {
  std::string export_path = GetEnvString("TL_SUNMMIO_ILP_PROBLEM_JSON");
  if (export_path.empty() && debug) {
    export_path = "body_ilp_problem.json";
  }
  if (!export_path.empty()) {
    WriteProblemJson(prob, export_path);
  }
}

bool IsPipelineLoopVar(const VarNode* node, const Var& pipeline_loop_var) {
  return pipeline_loop_var.defined() && node == pipeline_loop_var.get();
}

ffi::Map<Var, PrimExpr> BuildZeroSubstitutionMap(const PrimExpr& expr,
                                                 const Var& pipeline_loop_var) {
  std::unordered_set<const VarNode*> vars;
  PostOrderVisit(expr, [&](const ObjectRef& obj) {
    if (const auto* var = obj.as<VarNode>()) {
      if (!IsPipelineLoopVar(var, pipeline_loop_var)) {
        vars.insert(var);
      }
    }
  });

  ffi::Map<Var, PrimExpr> vmap;
  for (const VarNode* node : vars) {
    Var var = ffi::GetRef<Var>(node);
    vmap.Set(var, make_zero(var.dtype()));
  }
  return vmap;
}

int DetectIterOffsetFromExpr(const PrimExpr& expr, const Var& pipeline_loop_var,
                             arith::Analyzer* analyzer) {
  if (!pipeline_loop_var.defined() ||
      !UsesVar(expr, [v = pipeline_loop_var.get()](const VarNode* node) {
        return node == v;
      })) {
    return 0;
  }

  PrimExpr loop_only = expr;
  ffi::Map<Var, PrimExpr> vmap = BuildZeroSubstitutionMap(expr, pipeline_loop_var);
  if (!vmap.empty()) {
    loop_only = tir::Substitute(loop_only, vmap);
  }
  loop_only = analyzer->Simplify(loop_only);

  ffi::Array<PrimExpr> coeffs =
      arith::DetectLinearEquation(loop_only, ffi::Array<Var>{pipeline_loop_var});
  if (coeffs.size() != 2) {
    return 0;
  }

  PrimExpr coeff = analyzer->Simplify(coeffs[0]);
  PrimExpr base = analyzer->Simplify(coeffs[1]);
  const auto* coeff_int = coeff.as<IntImmNode>();
  if (coeff_int == nullptr || coeff_int->value == 0) {
    return 0;
  }

  PrimExpr unit_stride = coeff;
  PrimExpr offset_expr = analyzer->Simplify(floordiv(base, unit_stride));
  PrimExpr remainder = analyzer->Simplify(floormod(base, unit_stride));
  if (!analyzer->CanProveEqual(remainder, make_zero(remainder.dtype()))) {
    return 0;
  }

  const auto* offset_int = offset_expr.as<IntImmNode>();
  if (offset_int == nullptr) {
    return 0;
  }
  return static_cast<int>(offset_int->value);
}

int DetectIterOffsetFromRegion(const BufferRegion& region, const Var& pipeline_loop_var,
                               arith::Analyzer* analyzer) {
  int result = 0;
  bool found = false;
  for (const Range& range : region->region) {
    int dim_offset = DetectIterOffsetFromExpr(range->min, pipeline_loop_var, analyzer);
    bool uses_loop_var =
        pipeline_loop_var.defined() &&
        UsesVar(range->min, [v = pipeline_loop_var.get()](const VarNode* node) {
          return node == v;
        });
    if (!uses_loop_var) {
      continue;
    }
    if (!found) {
      result = dim_offset;
      found = true;
    } else if (result != dim_offset) {
      return 0;
    }
  }
  return found ? result : 0;
}

void DedupAccesses(std::vector<AccessInfo>* accesses) {
  std::vector<AccessInfo> deduped;
  deduped.reserve(accesses->size());
  for (const AccessInfo& access : *accesses) {
    bool exists = false;
    for (const AccessInfo& old : deduped) {
      if (access.is_write != old.is_write ||
          access.iter_offset != old.iter_offset ||
          !access.region->buffer.same_as(old.region->buffer)) {
        continue;
      }
      if (StructuralEqual()(access.region, old.region)) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      deduped.push_back(access);
    }
  }
  *accesses = std::move(deduped);
}

}  // namespace

class SunmmioStmtAccessAnalyzer : public StmtExprVisitor {
 public:
  explicit SunmmioStmtAccessAnalyzer(const PrimFunc& f) : analyzer_() {
    for (const auto& kv : f->buffer_map) {
      buffer_data_to_buffer_.Set(kv.second->data, kv.second);
    }
  }

  std::vector<AccessInfo> Collect(const Stmt& stmt,
                                  const Var& pipeline_loop_var = Var()) {
    accesses_.clear();
    pipeline_loop_var_ = pipeline_loop_var;
    VisitStmt(stmt);
    DedupAccesses(&accesses_);
    return accesses_;
  }

 private:
  void AddAccess(const BufferRegion& region, bool is_write) {
    accesses_.push_back(
        AccessInfo{region, is_write,
                   DetectIterOffsetFromRegion(region, pipeline_loop_var_, &analyzer_)});
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    Array<Range> region;
    for (const PrimExpr& index : op->indices) {
      region.push_back(Range::FromMinExtent(index, 1));
    }
    AddAccess(BufferRegion(op->buffer, region), true);
    VisitExpr(op->value);
  }

  void VisitStmt_(const EvaluateNode* op) final {
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(dma_copy())) {
        AddAccess(NormalizeToBufferRegion(call->args[0]), false);
        AddAccess(NormalizeToBufferRegion(call->args[1]), true);
        return;
      }
      if (call->op.same_as(mma_sunmmio())) {
        AddAccess(NormalizeToBufferRegion(call->args[0]), false);
        AddAccess(NormalizeToBufferRegion(call->args[1]), false);
        BufferRegion c_region = NormalizeToBufferRegion(call->args[2]);
        AddAccess(c_region, false);
        AddAccess(c_region, true);
        return;
      }
    }
    VisitExpr(op->value);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    Array<Range> region;
    for (const PrimExpr& index : op->indices) {
      if (const auto* ramp = index.as<RampNode>()) {
        region.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        region.push_back(Range::FromMinExtent(index, 1));
      }
    }
    AddAccess(BufferRegion(op->buffer, region), false);
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(RegionOp::Get())) {
      AddAccess(
          NormalizeToBufferRegion(ffi::GetRef<PrimExpr>(op)),
          false);
      return;
    }

    if (op->op.same_as(builtin::address_of())) {
      if (const auto* load = op->args[0].as<BufferLoadNode>()) {
        AddAccess(BufferRegion::FullRegion(load->buffer), false);
        return;
      }
      if (const auto* var_node = op->args[0].as<VarNode>()) {
        Var data_var = ffi::GetRef<Var>(var_node);
        auto it = buffer_data_to_buffer_.find(data_var);
        if (it != buffer_data_to_buffer_.end()) {
          AddAccess(BufferRegion::FullRegion((*it).second), false);
          return;
        }
      }
    }

    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (const auto* buffer_var = op->args[1].as<VarNode>()) {
        auto it = buffer_data_to_buffer_.find(ffi::GetRef<Var>(buffer_var));
        if (it != buffer_data_to_buffer_.end()) {
          AddAccess(BufferRegion::FullRegion((*it).second), false);
          return;
        }
      }
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  arith::Analyzer analyzer_;
  ffi::Map<Var, Buffer> buffer_data_to_buffer_;
  Var pipeline_loop_var_;
  std::vector<AccessInfo> accesses_;
};

class SunmmioRoleMarker : public StmtVisitor {
 public:
  SunmmioRoleMarker(ASTTraverser& traverser, const PrimFunc& func)
      : traverser_(traverser), access_analyzer_(func) {
    traverser_.clear();
  }

  Role GetRole(const StmtNode* stmt) const {
    auto it = map_.find(stmt);
    ICHECK(it != map_.end()) << "Cannot find role for stmt: "
                             << stmt->GetTypeKey();
    return it->second;
  }

  Role GetRole(const Stmt& stmt) const { return GetRole(stmt.get()); }

  std::vector<AccessInfo> GetAccesses(const Stmt& stmt,
                                      const Var& pipeline_loop_var = Var()) {
    return access_analyzer_.Collect(stmt, pipeline_loop_var);
  }

  void VisitStmt_(const EvaluateNode* op) final {
    Role role = Role::kConsumer;
    if (const auto* call = op->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
        if (IsGlobalBuffer(src_region->buffer)) {
          role = Role::kProducer;
        }
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    Role role = Role::kProducer;
    // Reuse the legacy traverser path for role classification. It is less
    // detailed than the ILP access collector but has proven stable on large
    // kernels such as flash-attention.
    traverser_.traverse_stmt(ffi::GetRef<Stmt>(op));
    auto reads = traverser_.read_buffer_regions_;
    for (const BufferRegion& read : reads) {
      if (!IsGlobalBuffer(read->buffer)) {
        role = Role::kConsumer;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->seq[0]);
    for (const Stmt& stmt : op->seq) {
      if (role != GetRole(stmt)) {
        role = Role::kBoth;
        break;
      }
    }
    SetRole(op, role);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetRole(op->then_case);
    if (op->else_case.defined() && role != GetRole(op->else_case.value())) {
      role = Role::kBoth;
    }
    SetRole(op, role);
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->block));
  }

  template <class NodeType>
  void HandleBodyStmt(const NodeType* op) {
    StmtVisitor::VisitStmt_(op);
    SetRole(op, GetRole(op->body));
  }

  void VisitStmt_(const ForNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AllocateNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const DeclBufferNode* op) final { HandleBodyStmt(op); }

 private:
  void SetRole(const StmtNode* stmt, Role role) { map_[stmt] = role; }

  std::unordered_map<const StmtNode*, Role> map_;
  ASTTraverser& traverser_;
  SunmmioStmtAccessAnalyzer access_analyzer_;
};

class SunmmioExprAnalyzer : public StmtExprVisitor {
 public:
  SunmmioExprAnalyzer() {}

  void Analyze(const PrimExpr& expr) {
    loop_cost_ = 0;
    load_times = 0;
    flops_ = 0;
    args_.clear();
    constants_.clear();
    vars_.clear();
    StmtExprVisitor::VisitExpr(expr);
  }

 private:
  void VisitExpr_(const MulNode* op) final {
    auto a = op->a;
    auto b = op->b;
    flops_ += 1;
    if (const auto* a_int = a.as<IntImmNode>()) {
      if (const auto* b_int = b.as<IntImmNode>()) {
        return;
      }
      if (a_int->value <= 32) {
        loop_cost_ += 2;
        StmtExprVisitor::VisitExpr(op->b);
        return;
      }
    }
    if (const auto* b_int = b.as<IntImmNode>()) {
      if (b_int->value <= 32) {
        loop_cost_ += 2;
        StmtExprVisitor::VisitExpr(op->a);
        return;
      }
    }
    loop_cost_ += 4;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const SubNode* op) final {
    loop_cost_ += 4;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const AddNode* op) final {
    loop_cost_ += 4;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MaxNode* op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const MinNode* op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const CastNode* op) final {
    loop_cost_ += 3;
    StmtExprVisitor::VisitExpr(op->value);
  }

  void VisitExpr_(const IntImmNode* op) final {
    bool insert = true;
    for (auto it : constants_) {
      if (ExprDeepEqual()(it, tvm::ffi::GetRef<PrimExpr>(op))) {
        insert = false;
        break;
      }
    }
    if (insert) {
      constants_.push_back(tvm::ffi::GetRef<PrimExpr>(op));
    }
  }

  void VisitExpr_(const FloatImmNode* op) final {
    bool insert = true;
    for (auto it : constants_) {
      if (ExprDeepEqual()(it, tvm::ffi::GetRef<PrimExpr>(op))) {
        insert = false;
        break;
      }
    }
    if (insert) {
      constants_.push_back(tvm::ffi::GetRef<PrimExpr>(op));
    }
  }

  void VisitExpr_(const VarNode* op) final {
    bool insert = true;
    for (auto it : vars_) {
      if (ExprDeepEqual()(it, tvm::ffi::GetRef<PrimExpr>(op))) {
        insert = false;
        break;
      }
    }
    if (insert) {
      vars_.push_back(tvm::ffi::GetRef<PrimExpr>(op));
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(Op::Get("tir.exp2"))) {
      loop_cost_ += 10;
      flops_ += 3;
      StmtExprVisitor::VisitExpr(op->args[0]);
    } else if (op->op.same_as(Op::Get("tl.infinity"))) {
      bool insert = true;
      for (auto it : constants_) {
        if (ExprDeepEqual()(it, FloatImm(DataType::Float(16),
                                         std::numeric_limits<float>::infinity()))) {
          insert = false;
          break;
        }
      }
      if (insert) {
        constants_.push_back(
            FloatImm(DataType::Float(16), std::numeric_limits<float>::infinity()));
      }
    } else if (op->op.same_as(Op::Get("tir.if_then_else"))) {
      bool insert = true;
      for (auto it : args_) {
        if (ExprDeepEqual()(it, op->args[0])) {
          insert = false;
          break;
        }
      }
      if (insert) {
        args_.push_back(op->args[0]);
      }
      StmtExprVisitor::VisitExpr(op->args[0]);
      StmtExprVisitor::VisitExpr(op->args[1]);
      StmtExprVisitor::VisitExpr(op->args[2]);
    } else if (op->op.same_as(Op::Get("tir.bitwise_and"))) {
      bool insert = true;
      for (auto it : args_) {
        if (ExprDeepEqual()(it, op->args[0])) {
          insert = false;
          break;
        }
      }
      if (insert) {
        args_.push_back(op->args[0]);
      }
      flops_ += 1;
      StmtExprVisitor::VisitExpr(op->args[0]);
      StmtExprVisitor::VisitExpr(op->args[1]);
    } else {
      ICHECK(0) << "Op " << op->op << " not supported now.";
    }
  }

  void VisitExpr_(const LENode* op) final {
    loop_cost_ += 3;
    flops_ += 1;
    StmtExprVisitor::VisitExpr(op->a);
    StmtExprVisitor::VisitExpr(op->b);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (load_times == 0) {
      load_times++;
      loop_cost_ += 14;
      flops_ += 5;
    } else {
      load_times++;
      loop_cost_ += 1;
      flops_ += 1;
    }
    for (auto arg : op->indices) {
      bool insert = true;
      for (auto it : args_) {
        if (ExprDeepEqual()(it, arg)) {
          insert = false;
          break;
        }
      }
      if (insert) {
        args_.push_back(arg);
      }
    }
  }

 public:
  float loop_cost_ = 0;
  Array<PrimExpr> args_;
  Array<PrimExpr> vars_;
  Array<PrimExpr> constants_;
  int load_times = 0;
  float flops_ = 0;
};

class TemplateCommand {
 public:
  int id{-1};
  std::string name;
  Stmt stmt;
  Role role{Role::kUndefined};
  DeviceType type{DeviceType::Unspecified};
  std::vector<AccessInfo> accesses;
  CommandSpec spec;

  TemplateCommand(int id, const Stmt& stmt)
      : id(id), name("cmd_" + std::to_string(id)), stmt(stmt) {}
};

DeviceType DetectDeviceType(const Stmt& stmt) {
  if (const auto* block = stmt.as<BlockRealizeNode>()) {
    if (const auto* eval = block->block->body.as<EvaluateNode>()) {
      if (const auto* call = eval->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.mma_sunmmio"))) {
          return DeviceType::TensorCore;
        }
      }
    }
    return DeviceType::VectorCore;
  }
  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        return DeviceType::ODMA;
      }
    }
  }
  return DeviceType::VectorCore;
}

std::vector<int> BuildIlpResources(const Stmt& stmt, DeviceType type,
                                   const std::vector<AccessInfo>& accesses) {
  std::vector<int> resources;
  auto add_resource = [&](int resource) {
    if (std::find(resources.begin(), resources.end(), resource) == resources.end()) {
      resources.push_back(resource);
    }
  };

  if (type == DeviceType::TensorCore) {
    add_resource(static_cast<int>(IlpResourceType::kTensorCore));
  } else if (type == DeviceType::VectorCore) {
    add_resource(static_cast<int>(IlpResourceType::kVectorCore));
  }

  if (const auto* eval = stmt.as<EvaluateNode>()) {
    if (const auto* call = eval->value.as<CallNode>()) {
      if (call->op.same_as(Op::Get("tl.dma_copy"))) {
        BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
        BufferRegion dst_region = NormalizeToBufferRegion(call->args[1]);
        if (IsGlobalBuffer(src_region->buffer)) {
          if (dst_region->buffer.scope() == "shared.asram") {
            LOG(FATAL) << "ILP graph does not model DRAM -> ASRAM dma path yet.";
          }
          if (dst_region->buffer.scope() == "shared.wsram") {
            add_resource(static_cast<int>(IlpResourceType::kODMA0));
            add_resource(static_cast<int>(IlpResourceType::kWsramIn));
            return resources;
          }
          if (dst_region->buffer.scope() == "shared.rsram" ||
              dst_region->buffer.scope() == "local") {
            add_resource(static_cast<int>(IlpResourceType::kODMA0));
            add_resource(static_cast<int>(IlpResourceType::kRsram));
            return resources;
          }
        }
        add_resource(static_cast<int>(IlpResourceType::kODMA0));
      }
    }
  }

  for (const AccessInfo& access : accesses) {
    const Buffer& buffer = access.buffer();
    if (IsGlobalBuffer(buffer)) {
      continue;
    }
    if (buffer.scope() == "shared.wsram") {
      add_resource(static_cast<int>(
          access.is_write ? IlpResourceType::kWsramIn : IlpResourceType::kWsramOut));
    } else if (buffer.scope() == "shared.asram") {
      add_resource(static_cast<int>(
          access.is_write ? IlpResourceType::kAsramIn : IlpResourceType::kAsramOut));
    } else if (buffer.scope() == "shared.rsram" || buffer.scope() == "local") {
      add_resource(static_cast<int>(IlpResourceType::kRsram));
    }
  }
  return resources;
}

int GetPingPongMemoryKind(const Buffer& buffer) {
  if (buffer.scope() == "shared.wsram") {
    return 0;
  }
  if (buffer.scope() == "shared.asram") {
    return 1;
  }
  return -1;
}

int GetMemoryWriteResource(int mem) {
  if (mem == 0) {
    return static_cast<int>(IlpResourceType::kWsramIn);
  }
  if (mem == 1) {
    return static_cast<int>(IlpResourceType::kAsramIn);
  }
  return -1;
}

int GetMemoryReadResource(int mem) {
  if (mem == 0) {
    return static_cast<int>(IlpResourceType::kWsramOut);
  }
  if (mem == 1) {
    return static_cast<int>(IlpResourceType::kAsramOut);
  }
  return -1;
}

bool CommandUsesResource(const CommandSpec& spec, int resource) {
  return std::find(spec.resources.begin(), spec.resources.end(), resource) !=
         spec.resources.end();
}

FlowSpec MakeInternalFlowSpec(const Problem& problem, int prod, int cons, int mem) {
  FlowSpec flow;
  flow.resident = false;
  flow.prod = prod;
  flow.cons = cons;
  flow.mem = mem;
  flow.fixed_bank = -1;
  // Keep the initial ILP input coarse-grained: each SRAM flow currently counts
  // as one bank-capacity unit until a more precise footprint model is wired in.
  flow.fp = 1;
  flow.initial_time = 0;
  flow.w_off = 0;
  flow.w_dur = problem.P[prod].latency;
  flow.r_off = 0;
  flow.r_dur = problem.P[cons].latency;
  flow.write_resource = GetMemoryWriteResource(mem);
  flow.read_resource = GetMemoryReadResource(mem);
  return flow;
}

int EstimateCommandLatency(const Stmt& stmt, DeviceType type) {
  if (type == DeviceType::TensorCore) {
    if (const auto* block = stmt.as<BlockRealizeNode>()) {
      auto body = block->block->body;
      if (const auto* eval = body.as<EvaluateNode>()) {
        if (const auto* call = eval->value.as<CallNode>()) {
          if (call->op.same_as(Op::Get("tl.mma_sunmmio"))) {
            auto A = call->args[0].as<CallNode>();
            auto B = call->args[1].as<CallNode>();
            auto row_size = A->args[2].as<IntImmNode>()->value;
            auto col_size = B->args[3].as<IntImmNode>()->value;
            auto acc_size = A->args[3].as<IntImmNode>()->value;
            int min_blk_num = 0;
            if (row_size <= 16 && col_size <= 32) {
              min_blk_num = 1;
            } else {
              min_blk_num = 4;
            }
            int gap = std::max(CeilDiv(acc_size, 32), min_blk_num);
            return 11 + 5 + 5 + 8 +
                   CeilDiv(row_size, 16) * CeilDiv(col_size, 32) * gap + 70;
          }
        }
      }
    }
    ICHECK(0) << "Can't identify TensorCore delay for command " << stmt;
  }
  if (type == DeviceType::ODMA) {
    if (const auto* eval = stmt.as<EvaluateNode>()) {
      if (const auto* call = eval->value.as<CallNode>()) {
        if (call->op.same_as(Op::Get("tl.dma_copy"))) {
          auto src = call->args[0].as<CallNode>();
          int nums = 1;
          for (int i = 2; i < static_cast<int>(src->args.size()); ++i) {
            nums *= src->args[i].as<IntImmNode>()->value;
          }
          int num_kb = CeilDiv(nums, 512);
          return 50 + num_kb;
        }
      }
    }
    ICHECK(0) << "Can't identify ODMA delay for command " << stmt;
  }
  if (type == DeviceType::VectorCore) {
    float flops_cost = 0;
    if (const auto* for_node = stmt.as<ForNode>()) {
      PrimExpr extent = for_node->extent;
      auto current = for_node;
      while (current->body.as<ForNode>()) {
        current = current->body.as<ForNode>();
        extent *= current->extent;
      }
      if (const auto* store = current->body.as<BufferStoreNode>()) {
        SunmmioExprAnalyzer analyzer;
        analyzer.Analyze(store->value);
        flops_cost += (5 + analyzer.flops_) * (extent.as<IntImmNode>()->value);
        int iterations = CeilDiv((extent.as<IntImmNode>()->value) * 16, 4096);
        for (auto& it : analyzer.constants_) {
          flops_cost += 1;
        }
        auto args = analyzer.args_;
        Array<PrimExpr> index_vars;
        for (auto& it : args) {
          index_vars.push_back(it);
        }
        for (auto& it : store->indices) {
          bool insert = true;
          for (auto& var : index_vars) {
            if (ExprDeepEqual()(var, it)) {
              insert = false;
              break;
            }
          }
          if (insert) {
            index_vars.push_back(it);
          }
        }
        for (auto& it : index_vars) {
          analyzer.Analyze(it);
          flops_cost += analyzer.flops_ * (extent.as<IntImmNode>()->value);
        }
        flops_cost = CeilDiv(static_cast<int>(flops_cost), 250);
        return static_cast<int>(flops_cost);
      }
      ICHECK(0) << "VectorCore for-loop command must end with BufferStore";
    } else if (const auto* block_node = stmt.as<BlockNode>()) {
      auto body = block_node->body;
      if (const auto* for_node = body.as<ForNode>()) {
        PrimExpr out_extent = 1;
        auto current = for_node;
        auto previous = current;
        while (current->body.as<ForNode>()) {
          out_extent *= current->extent;
          previous = current;
          current = current->body.as<ForNode>();
        }
        if (const auto seq = current->body.as<SeqStmtNode>()) {
          ICHECK(seq->seq.size() == 3) << "Error format of reduce op";
          if (const auto init_for = seq->seq[0].as<ForNode>()) {
            PrimExpr init_extent = init_for->extent;
            auto current_init_for = init_for;
            while (current_init_for->body.as<ForNode>()) {
              current_init_for = current_init_for->body.as<ForNode>();
              init_extent *= current_init_for->extent;
            }
            flops_cost += (5 + 1) * init_extent.as<IntImmNode>()->value;
          }
          if (const auto for_stmt = seq->seq[1].as<ForNode>()) {
            auto inner_for = seq->seq[1];
            PrimExpr inner_extent = 1;
            while (inner_for.as<ForNode>()) {
              inner_extent *= inner_for.as<ForNode>()->extent;
              inner_for = inner_for.as<ForNode>()->body;
            }
            ICHECK(inner_for.as<BufferStoreNode>()) << "Error format of reduce op";
            auto store = inner_for.as<BufferStoreNode>();
            SunmmioExprAnalyzer analyzer;
            analyzer.Analyze(store->value);
            flops_cost +=
                (5 + analyzer.flops_) *
                (inner_extent * previous->extent).as<IntImmNode>()->value;
          }
          if (const auto eval_stmt = seq->seq[2].as<EvaluateNode>()) {
            auto reduce = eval_stmt->value.as<CallNode>();
            if (reduce->args[0].as<StringImmNode>()->value == "max" ||
                reduce->args[0].as<StringImmNode>()->value == "min") {
              flops_cost += 1;
            } else if (reduce->args[0].as<StringImmNode>()->value == "sum") {
              flops_cost += 3;
            }
          }
          flops_cost =
              CeilDiv(static_cast<int>(flops_cost * out_extent.as<IntImmNode>()->value), 250);
          return static_cast<int>(flops_cost);
        }
      }
    }
    ICHECK(0) << "Unsupported VectorCore command " << stmt;
  }
  ICHECK(0) << "Unsupported command type for delay " << stmt;
}

int PositiveMod(int value, int mod) {
  if (mod <= 0) {
    return value;
  }
  int result = value % mod;
  if (result < 0) {
    result += mod;
  }
  return result;
}

BufferRegion MaterializeBufferRegion(const BufferRegion& region, const Var& loop_var,
                                     int iter) {
  if (!loop_var.defined()) {
    return region;
  }
  ffi::Map<Var, PrimExpr> vmap;
  vmap.Set(loop_var, make_const(loop_var.dtype(), iter));
  Array<Range> materialized;
  for (const Range& rng : region->region) {
    PrimExpr min = tir::Substitute(rng->min, vmap);
    PrimExpr extent = tir::Substitute(rng->extent, vmap);
    materialized.push_back(Range::FromMinExtent(min, extent));
  }
  return BufferRegion(region->buffer, materialized);
}

std::vector<Buffer> DetectVersionedBuffers(
    const std::vector<TemplateCommand>& commands) {
  std::set<Buffer> used_buffers;
  std::unordered_set<const BufferNode*> consumer_used;
  std::unordered_set<const BufferNode*> producer_used;
  std::unordered_map<const BufferNode*, int> first_write_index;
  std::unordered_map<const BufferNode*, std::vector<int>> write_indexes;
  std::unordered_map<const BufferNode*, int> first_read_index;
  std::unordered_map<const BufferNode*, int> last_read_index;
  std::vector<Buffer> versioned_buffers;

  auto is_copy_stage = [&](int idx) {
    bool has_shared_write = false;
    bool has_global_read = false;
    for (const AccessInfo& access : commands[idx].accesses) {
      if (access.is_write && IsSunmmioSharedBuffer(access.buffer())) {
        has_shared_write = true;
      }
      if (!access.is_write && IsGlobalBuffer(access.buffer())) {
        has_global_read = true;
      }
    }
    return has_shared_write && has_global_read;
  };

  for (int i = 0; i < static_cast<int>(commands.size()); ++i) {
    bool copy_stage = is_copy_stage(i);
    bool is_producer = commands[i].role == Role::kProducer ||
                       (commands[i].role == Role::kBoth && copy_stage);
    bool is_consumer = commands[i].role == Role::kConsumer ||
                       (commands[i].role == Role::kBoth && !copy_stage);
    for (const AccessInfo& access : commands[i].accesses) {
      if (IsGlobalBuffer(access.buffer())) {
        continue;
      }
      used_buffers.insert(access.buffer());
      const BufferNode* buf = access.buffer().get();
      if (access.is_write) {
        if (is_producer) {
          producer_used.insert(buf);
        }
        if (!first_write_index.count(buf)) {
          first_write_index[buf] = i;
        }
        write_indexes[buf].push_back(i);
      } else {
        if (is_consumer) {
          consumer_used.insert(buf);
        }
        if (!first_read_index.count(buf)) {
          first_read_index[buf] = i;
        }
        last_read_index[buf] = i;
      }
    }
  }

  for (const Buffer& buffer : used_buffers) {
    const BufferNode* buf = buffer.get();
    if (consumer_used.count(buf) && producer_used.count(buf)) {
      auto r = first_read_index.find(buf);
      auto w = first_write_index.find(buf);
      if (r != first_read_index.end() && w != first_write_index.end() && r->second > w->second) {
        versioned_buffers.push_back(buffer);
        continue;
      }
    }
    auto it_w = first_write_index.find(buf);
    auto it_r = last_read_index.find(buf);
    if (it_w != first_write_index.end() && it_r != last_read_index.end() &&
        it_w->second < it_r->second && is_copy_stage(it_w->second)) {
      versioned_buffers.push_back(buffer);
    }
  }

  bool updated = true;
  while (updated) {
    updated = false;
    for (const Buffer& buffer : used_buffers) {
      if (std::find(versioned_buffers.begin(), versioned_buffers.end(), buffer) !=
          versioned_buffers.end()) {
        continue;
      }
      const BufferNode* buf = buffer.get();
      auto it_writes = write_indexes.find(buf);
      auto it_first_w = first_write_index.find(buf);
      auto it_first_r = first_read_index.find(buf);
      if (it_writes == write_indexes.end() || it_writes->second.empty() ||
          it_first_w == first_write_index.end() || it_first_r == first_read_index.end()) {
        continue;
      }
      bool can_propagate = it_first_w->second < it_first_r->second;
      for (int idx : it_writes->second) {
        for (const AccessInfo& access : commands[idx].accesses) {
          if (access.is_write || IsGlobalBuffer(access.buffer())) {
            continue;
          }
          if (std::find(versioned_buffers.begin(), versioned_buffers.end(), access.buffer()) ==
              versioned_buffers.end()) {
            can_propagate = false;
            break;
          }
        }
        if (!can_propagate) {
          break;
        }
      }
      if (can_propagate) {
        versioned_buffers.push_back(buffer);
        updated = true;
      }
    }
  }
  return versioned_buffers;
}

void BuildTemplateDependencyGraph(
    const std::vector<TemplateCommand>& commands, int iter_mod,
    const std::vector<Buffer>& versioned_buffers, const Var& pipeline_loop_var,
    Problem* problem) {
  std::unordered_set<const BufferNode*> versioned;
  for (const Buffer& buffer : versioned_buffers) {
    versioned.insert(buffer.get());
  }

  struct ExpandedCommandInstance {
    int template_id{-1};
    int iter{-1};
  };
  struct ExpandedAccessRecord {
    BufferRegion region;
    int cmd_idx{-1};
    int access_idx{-1};
    bool is_write{false};
    int logical_iter{0};
  };

  int dag_iterations = std::max(2, iter_mod + 1);
  std::vector<ExpandedCommandInstance> expanded;
  expanded.reserve(commands.size() * dag_iterations);
  for (int iter = 0; iter < dag_iterations; ++iter) {
    for (const TemplateCommand& cmd : commands) {
      expanded.push_back({cmd.id, iter});
    }
  }

  std::vector<std::vector<int>> predecessors(expanded.size());
  std::unordered_map<const BufferNode*, std::vector<ExpandedAccessRecord>> history;
  problem->flows.clear();
  std::set<std::tuple<int, int, const BufferNode*, int, int>> seen_flow_keys;
  for (int curr_idx = 0; curr_idx < static_cast<int>(expanded.size()); ++curr_idx) {
    const ExpandedCommandInstance& curr = expanded[curr_idx];
    const TemplateCommand& curr_cmd = commands[curr.template_id];

    for (int curr_access_idx = 0; curr_access_idx < static_cast<int>(curr_cmd.accesses.size());
         ++curr_access_idx) {
      const AccessInfo& curr_access_template = curr_cmd.accesses[curr_access_idx];
      BufferRegion curr_region =
          MaterializeBufferRegion(curr_access_template.region, pipeline_loop_var, curr.iter);
      int curr_logical_iter = curr.iter + curr_access_template.iter_offset;
      const BufferNode* buf = curr_region->buffer.get();
      int flow_mem = GetPingPongMemoryKind(curr_region->buffer);
      auto it_hist = history.find(buf);
      if (it_hist == history.end()) {
        history[buf].push_back({curr_region, curr_idx, curr_access_idx,
                                curr_access_template.is_write, curr_logical_iter});
        continue;
      }

      auto* entries = &it_hist->second;
      if (!curr_access_template.is_write) {
        for (auto it = entries->rbegin(); it != entries->rend(); ++it) {
          if (!it->is_write) {
            continue;
          }
          if (versioned.count(buf) &&
              PositiveMod(it->logical_iter, iter_mod) !=
                  PositiveMod(curr_logical_iter, iter_mod)) {
            continue;
          }
          if (!PipelineRegionIntersect(curr_region->region, it->region->region)) {
            continue;
          }
          if (std::find(predecessors[curr_idx].begin(), predecessors[curr_idx].end(),
                        it->cmd_idx) == predecessors[curr_idx].end()) {
            predecessors[curr_idx].push_back(it->cmd_idx);
          }
          if (flow_mem >= 0) {
            const ExpandedCommandInstance& src = expanded[it->cmd_idx];
            int write_resource = GetMemoryWriteResource(flow_mem);
            int read_resource = GetMemoryReadResource(flow_mem);
            if (src.template_id != curr.template_id &&
                CommandUsesResource(problem->P[src.template_id], write_resource) &&
                CommandUsesResource(problem->P[curr.template_id], read_resource)) {
              auto flow_key = std::make_tuple(src.template_id, curr.template_id, buf,
                                              it->access_idx, curr_access_idx);
              if (seen_flow_keys.insert(flow_key).second) {
                problem->flows.push_back(
                    MakeInternalFlowSpec(*problem, src.template_id, curr.template_id, flow_mem));
              }
            }
          }
          break;
        }
      } else {
        for (auto it = entries->rbegin(); it != entries->rend(); ++it) {
          if (versioned.count(buf) &&
              PositiveMod(it->logical_iter, iter_mod) !=
                  PositiveMod(curr_logical_iter, iter_mod)) {
            continue;
          }
          if (!PipelineRegionIntersect(curr_region->region, it->region->region)) {
            continue;
          }
          if (std::find(predecessors[curr_idx].begin(), predecessors[curr_idx].end(),
                        it->cmd_idx) == predecessors[curr_idx].end()) {
            predecessors[curr_idx].push_back(it->cmd_idx);
          }
          if (it->is_write) {
            break;
          }
        }
      }

      history[buf].push_back(
          {curr_region, curr_idx, curr_access_idx, curr_access_template.is_write,
           curr_logical_iter});
    }
  }

  std::map<std::pair<int, int>, int> best_delta;
  problem->dep_edges.clear();
  problem->delta.clear();
  for (int dst_idx = 0; dst_idx < static_cast<int>(expanded.size()); ++dst_idx) {
    const ExpandedCommandInstance& dst = expanded[dst_idx];
    for (int src_idx : predecessors[dst_idx]) {
      const ExpandedCommandInstance& src = expanded[src_idx];
      std::pair<int, int> key{src.template_id, dst.template_id};
      int delta = dst.iter - src.iter;
      auto it = best_delta.find(key);
      if (it == best_delta.end() || delta < it->second) {
        best_delta[key] = delta;
      }
    }
  }
  for (const auto& it : best_delta) {
    problem->dep_edges.push_back(it.first);
    problem->delta[EdgeKey(it.first.first, it.first.second)] = it.second;
  }
}

int ResourceLowerBound(const Problem& prob) {
  int lb = 1;
  for (int r : prob.R) {
    int cap = 1;
    auto it = prob.cap.find(r);
    if (it != prob.cap.end()) {
      cap = it->second;
    }
    if (cap <= 0) {
      continue;
    }
    long long total = 0;
    for (int i = 0; i < prob.N; ++i) {
      if (std::find(prob.P[i].resources.begin(), prob.P[i].resources.end(), r) !=
          prob.P[i].resources.end()) {
        total += prob.P[i].latency;
      }
    }
    lb = std::max(lb, std::max(1, int((total + cap - 1) / cap)));
  }
  return lb;
}

HighsInt AddCol(Highs& highs, double lower, double upper, double cost,
                bool is_integer) {
  HighsStatus st = highs.addCol(cost, lower, upper, 0, nullptr, nullptr);
  ICHECK(st == HighsStatus::kOk) << "addCol failed";
  HighsInt col = highs.getNumCol() - 1;
  if (is_integer) {
    highs.changeColIntegrality(col, HighsVarType::kInteger);
  }
  return col;
}

HighsStatus AddRow(Highs& highs, double lower, double upper,
                   const std::vector<HighsInt>& idx,
                   const std::vector<double>& val) {
  const HighsInt* idx_ptr = idx.empty() ? nullptr : idx.data();
  const double* val_ptr = val.empty() ? nullptr : val.data();
  return highs.addRow(lower, upper, HighsInt(idx.size()), idx_ptr, val_ptr);
}

void MergeLinearTerms(const std::vector<HighsInt>& idx,
                      const std::vector<double>& val,
                      std::vector<HighsInt>& merged_idx,
                      std::vector<double>& merged_val) {
  std::map<HighsInt, double> acc;
  for (size_t k = 0; k < idx.size(); ++k) {
    acc[idx[k]] += val[k];
  }
  merged_idx.clear();
  merged_val.clear();
  for (const auto& kv : acc) {
    if (kv.second == 0.0) {
      continue;
    }
    merged_idx.push_back(kv.first);
    merged_val.push_back(kv.second);
  }
}

void AddLeq(Highs& highs, const std::vector<HighsInt>& idx,
            const std::vector<double>& val, double rhs) {
  std::vector<HighsInt> merged_idx;
  std::vector<double> merged_val;
  MergeLinearTerms(idx, val, merged_idx, merged_val);
  if (merged_idx.empty()) {
    if (0.0 > rhs) {
      AddRow(highs, 1.0, 0.0, {}, {});
    }
    return;
  }
  AddRow(highs, -kInf, rhs, merged_idx, merged_val);
}

void AddEq(Highs& highs, const std::vector<HighsInt>& idx,
           const std::vector<double>& val, double rhs) {
  std::vector<HighsInt> merged_idx;
  std::vector<double> merged_val;
  MergeLinearTerms(idx, val, merged_idx, merged_val);
  AddRow(highs, rhs, rhs, merged_idx, merged_val);
}

void BuildVectorIssueGap(Highs& highs, const Problem& prob, int II,
                         const ModelVars& vars) {
  for (int i = 0; i < prob.N; ++i) {
    if (std::find(prob.P[i].resources.begin(), prob.P[i].resources.end(),
                  static_cast<int>(IlpResourceType::kVectorCore)) ==
        prob.P[i].resources.end()) {
      continue;
    }
    int dur = prob.P[i].latency;
    for (int s = 0; s < II; ++s) {
      std::vector<HighsInt> idx;
      std::vector<double> val;
      for (int k = 0; k < prob.N; ++k) {
        idx.push_back(vars.col_x[k][s]);
        val.push_back(1.0);
      }
      for (int st = 0; st < II; ++st) {
        bool blocked = false;
        for (int u = st + 1; u < st + dur; ++u) {
          if (u % II == s) {
            blocked = true;
            break;
          }
        }
        if (blocked) {
          idx.push_back(vars.col_x[i][st]);
          val.push_back(double(prob.N));
        }
      }
      AddLeq(highs, idx, val, double(prob.N));
    }
  }
}

ModelVars BuildModel(Highs& highs, const Problem& prob, int II, bool optimize_t,
                     int threads) {
  highs.clear();
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("log_to_console", false);
  highs.setOptionValue("threads", threads);
  highs.setOptionValue("parallel", "on");
  highs.changeObjectiveSense(ObjSense::kMinimize);

  int max_delta = 0;
  int max_latency = 0;
  for (const auto& kv : prob.delta) {
    max_delta = std::max(max_delta, kv.second);
  }
  for (const auto& spec : prob.P) {
    max_latency = std::max(max_latency, spec.latency);
  }
  int time_ub = prob.Tmax + max_delta * II + max_latency;

  ModelVars vars;
  vars.col_t.resize(prob.N);
  vars.col_y.resize(prob.N);
  vars.col_m.resize(prob.N);
  vars.col_x.assign(prob.N, std::vector<HighsInt>(II, -1));
  vars.col_a.assign(prob.N, std::vector<HighsInt>(II, -1));

  for (int i = 0; i < prob.N; ++i) {
    vars.col_t[i] = AddCol(highs, 0, time_ub, 0, true);
    vars.col_y[i] = AddCol(highs, 0, time_ub, 0, true);
    vars.col_m[i] = AddCol(highs, 0, II - 1, 0, true);
  }
  for (int i = 0; i < prob.N; ++i) {
    for (int s = 0; s < II; ++s) {
      vars.col_x[i][s] = AddCol(highs, 0, 1, 0, true);
    }
  }
  for (int i = 0; i < prob.N; ++i) {
    int ub = CeilDiv(prob.P[i].latency, II);
    for (int s = 0; s < II; ++s) {
      vars.col_a[i][s] = AddCol(highs, 0, ub, 0, true);
    }
  }
  vars.col_T = AddCol(highs, 0, time_ub, optimize_t ? 1.0 : 0.0, true);

  for (int i = 0; i < prob.N; ++i) {
    std::vector<HighsInt> idx1;
    std::vector<double> val1;
    for (int s = 0; s < II; ++s) {
      idx1.push_back(vars.col_x[i][s]);
      val1.push_back(1.0);
    }
    AddEq(highs, idx1, val1, 1.0);

    std::vector<HighsInt> idx2{vars.col_m[i]};
    std::vector<double> val2{1.0};
    for (int s = 0; s < II; ++s) {
      idx2.push_back(vars.col_x[i][s]);
      val2.push_back(-double(s));
    }
    AddEq(highs, idx2, val2, 0.0);

    AddEq(highs, {vars.col_t[i], vars.col_y[i], vars.col_m[i]},
          {1.0, -double(II), -1.0}, 0.0);
  }

  for (int i = 0; i < prob.N; ++i) {
    int dur = prob.P[i].latency;
    for (int s = 0; s < II; ++s) {
      std::vector<HighsInt> idx{vars.col_a[i][s]};
      std::vector<double> val{1.0};
      for (int st = 0; st < II; ++st) {
        int rel = (s - st) % II;
        if (rel < 0) {
          rel += II;
        }
        int cnt = 0;
        if (rel < dur) {
          cnt = CeilDiv(dur - rel, II);
        }
        if (cnt != 0) {
          idx.push_back(vars.col_x[i][st]);
          val.push_back(-double(cnt));
        }
      }
      AddEq(highs, idx, val, 0.0);
    }
  }

  for (const auto& e : prob.dep_edges) {
    int i = e.first;
    int j = e.second;
    int d = prob.P[i].latency;
    int delta = prob.delta.at(EdgeKey(i, j));
    if (i == j) {
      if (d > delta * II) {
        AddRow(highs, 1.0, 0.0, {}, {});
      }
      continue;
    }
    AddLeq(highs, {vars.col_t[i], vars.col_t[j]}, {1.0, -1.0},
           double(delta * II - d));
  }

  for (int i = 0; i < prob.N; ++i) {
    AddLeq(highs, {vars.col_t[i], vars.col_T}, {1.0, -1.0},
           double(-prob.P[i].latency));
  }

  BuildVectorIssueGap(highs, prob, II, vars);

  for (int r : prob.R) {
    int cap = prob.cap.count(r) ? prob.cap.at(r) : 1;
    for (int s = 0; s < II; ++s) {
      std::vector<HighsInt> idx;
      std::vector<double> val;
      for (int i = 0; i < prob.N; ++i) {
        if (std::find(prob.P[i].resources.begin(), prob.P[i].resources.end(), r) !=
            prob.P[i].resources.end()) {
          idx.push_back(vars.col_a[i][s]);
          val.push_back(1.0);
        }
      }
      AddLeq(highs, idx, val, double(cap));
    }
  }

  return vars;
}

SolveResult SolveFixedII(const Problem& prob, int II, bool optimize_t,
                         int threads) {
  Highs highs;
  ModelVars vars = BuildModel(highs, prob, II, optimize_t, threads);
  highs.run();
  if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
    return {};
  }

  const HighsSolution& sol = highs.getSolution();
  SolveResult res;
  res.ok = true;
  res.II = II;
  if (!optimize_t) {
    return res;
  }
  res.t.resize(prob.N);
  res.m.resize(prob.N);
  res.y.resize(prob.N);
  for (int i = 0; i < prob.N; ++i) {
    res.t[i] = int(std::llround(sol.col_value[vars.col_t[i]]));
    res.m[i] = int(std::llround(sol.col_value[vars.col_m[i]]));
    res.y[i] = int(std::llround(sol.col_value[vars.col_y[i]]));
  }
  res.makespan = int(std::llround(sol.col_value[vars.col_T]));
  return res;
}

SolveResult FindMinimalII(const Problem& prob, int threads) {
  int lb = std::max(1, ResourceLowerBound(prob));
  int l = std::max(1, lb - 5);
  int r = std::min(std::max(1, prob.Tmax), std::max(l, (3 * l + 1) / 2));
  int best_ii = -1;
  while (l <= r) {
    int mid = (l + r) / 2;
    SolveResult feas = SolveFixedII(prob, mid, false, threads);
    if (feas.ok) {
      best_ii = mid;
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  if (best_ii < 0) {
    return {};
  }
  return SolveFixedII(prob, best_ii, true, threads);
}

class SunmmioPipelinePlannerILP : public StmtExprMutator {
 public:
  static Stmt Substitute(const PrimFunc& f, bool debug) {
    SunmmioPipelinePlannerILP planner(f, debug);
    return planner.VisitStmt(f->body);
  }

 private:
  SunmmioPipelinePlannerILP(const PrimFunc& f, bool debug)
      : func_(f), traverser_(f), debug_(debug) {}

  Optional<For> FindPipelineLoop(const Stmt& stmt) {
    Optional<For> result;
    PostOrderVisit(stmt, [&](const ObjectRef& obj) {
      if (result.defined()) {
        return;
      }
      if (const auto* loop = obj.as<ForNode>()) {
        if (loop->annotations.find("num_stages") != loop->annotations.end()) {
          result = ffi::GetRef<For>(loop);
        }
      }
    });
    return result;
  }

  const SeqStmtNode* GetPipelineBodySeq(const For& loop) {
    Stmt current = loop->body;
    if (const auto* realize = current.as<BlockRealizeNode>()) {
      current = realize->block->body;
    }
    while (true) {
      if (const auto* seq = current.as<SeqStmtNode>()) {
        return seq;
      }
      if (const auto* if_node = current.as<IfThenElseNode>()) {
        ICHECK(!if_node->else_case.defined());
        current = if_node->then_case;
        continue;
      }
      if (const auto* let_node = current.as<LetStmtNode>()) {
        current = let_node->body;
        continue;
      }
      return nullptr;
    }
  }

  struct IlpLoopAnalysis {
    Problem prob;
    std::vector<TemplateCommand> commands;
    std::set<Buffer> used_buffers;
    int iterations{0};
  };

  IlpLoopAnalysis AnalyzeLoop(const For& loop,
                              const SeqStmtNode* pipeline_body_seq) {
    IlpLoopAnalysis result;
    int num_stages = -1;
    auto it = loop->annotations.find("num_stages");
    ICHECK(it != loop->annotations.end());
    const auto& any_ref = (*it).second;
    if (const auto* imm = any_ref.as<IntImmNode>()) {
      num_stages = imm->value;
    }
    ICHECK_GT(num_stages, 0);
    result.iterations = num_stages;

    SunmmioRoleMarker role_marker(traverser_, func_);
    result.commands.reserve(pipeline_body_seq->seq.size());

    std::set<int> resource_set;
    int total_latency = 0;
    for (int i = 0; i < static_cast<int>(pipeline_body_seq->seq.size()); ++i) {
      const Stmt& stmt = pipeline_body_seq->seq[i];
      TemplateCommand cmd(i, stmt);
      role_marker(stmt);
      cmd.role = role_marker.GetRole(stmt);
      cmd.type = DetectDeviceType(stmt);
      cmd.accesses = role_marker.GetAccesses(stmt, loop->loop_var);
      cmd.spec.latency = EstimateCommandLatency(stmt, cmd.type);
      cmd.spec.resources = BuildIlpResources(stmt, cmd.type, cmd.accesses);
      cmd.spec.name = cmd.name;
      total_latency += cmd.spec.latency;
      for (int resource : cmd.spec.resources) {
        resource_set.insert(resource);
      }
      for (const AccessInfo& access : cmd.accesses) {
        if (!IsGlobalBuffer(access.buffer())) {
          result.used_buffers.insert(access.buffer());
        }
      }
      result.commands.push_back(cmd);
    }

    result.prob.N = static_cast<int>(result.commands.size());
    result.prob.Tmax = total_latency + 10;
    result.prob.R.assign(resource_set.begin(), resource_set.end());
    for (int resource : result.prob.R) {
      result.prob.cap[resource] = 1;
    }
    result.prob.P.resize(result.prob.N);
    for (const TemplateCommand& cmd : result.commands) {
      result.prob.P[cmd.id] = cmd.spec;
    }
    BuildTemplateDependencyGraph(result.commands, num_stages, /*versioned_buffers=*/{},
                                 loop->loop_var, &result.prob);
    return result;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For loop = ffi::GetRef<For>(op);
    if (op->annotations.find("num_stages") == op->annotations.end()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    const SeqStmtNode* pipeline_body_seq = GetPipelineBodySeq(loop);
    ICHECK(pipeline_body_seq != nullptr) << "Pipeline body must normalize to SeqStmt.";
    IlpLoopAnalysis analysis = AnalyzeLoop(loop, pipeline_body_seq);
    MaybeExportProblemJson(analysis.prob, debug_);
    int threads = GetEnvInt("HIGHS_THREADS", 20);
    SolveResult sol = FindMinimalII(analysis.prob, threads);
    ICHECK(sol.ok) << "ILP solve failed for sunmmio pipeline planning.";
    if (debug_) {
      LOG(INFO) << "ILP problem N=" << analysis.prob.N
                << " dep_edges=" << analysis.prob.dep_edges.size()
                << " flows=" << analysis.prob.flows.size()
                << " solved=" << sol.ok
                << " ii=" << sol.II;
    }

    Map<String, Any> annotations;
    for (const auto& kv : op->annotations) {
      if (kv.first != "num_stages" && kv.first != "versioned_buffers") {
        annotations.Set(kv.first, kv.second);
      }
    }

    annotations.Set("iterations", Integer(analysis.iterations));
    Array<String> prologue_orders;
    Array<String> body_orders;
    Array<String> epilogue_orders;

    std::vector<int> producer_ids;
    std::vector<int> body_template_ids;
    for (const TemplateCommand& cmd : analysis.commands) {
      if (cmd.role == Role::kProducer) {
        producer_ids.push_back(cmd.id);
        prologue_orders.push_back(
            String(std::to_string(0) + "-" + std::to_string(cmd.id)));
      } else {
        body_template_ids.push_back(cmd.id);
      }
    }

    for (int iter = 0; iter < analysis.iterations; ++iter) {
      std::vector<int> iter_ids = body_template_ids;
      std::sort(iter_ids.begin(), iter_ids.end(), [&](int a, int b) {
        if (sol.t[a] != sol.t[b]) {
          return sol.t[a] < sol.t[b];
        }
        return a < b;
      });
      for (int id : iter_ids) {
        body_orders.push_back(
            String(std::to_string(iter) + "-" + std::to_string(id)));
      }

      std::vector<int> producer_sorted = producer_ids;
      std::sort(producer_sorted.begin(), producer_sorted.end(), [&](int a, int b) {
        if (sol.t[a] != sol.t[b]) {
          return sol.t[a] < sol.t[b];
        }
        return a < b;
      });
      for (int id : producer_sorted) {
        body_orders.push_back(
            String(std::to_string(iter + 1) + "-" + std::to_string(id)));
      }
    }

    int epilogue_iterations = -1;
    PrimExpr epilogue_iterations_expr = floormod(loop->extent, analysis.iterations);
    if (const auto* mod_int = epilogue_iterations_expr.as<IntImmNode>()) {
      epilogue_iterations = mod_int->value;
    }
    ICHECK(epilogue_iterations != -1)
        << "Can't calculate epilogue iterations for ILP pipeline planning.";
    if (epilogue_iterations == 0) {
      epilogue_iterations = analysis.iterations;
    }

    int epilogue_iter = 0;
    for (int i = 0; i < epilogue_iterations - 1; ++i) {
      std::vector<int> all_ids(analysis.prob.N);
      for (int j = 0; j < analysis.prob.N; ++j) {
        all_ids[j] = j;
      }
      std::sort(all_ids.begin(), all_ids.end(), [&](int a, int b) {
        if (sol.t[a] != sol.t[b]) {
          return sol.t[a] < sol.t[b];
        }
        return a < b;
      });
      for (int id : all_ids) {
        epilogue_orders.push_back(
            String(std::to_string(epilogue_iter) + "-" + std::to_string(id)));
      }
      epilogue_iter += 1;
    }
    std::vector<int> tail_ids = body_template_ids;
    std::sort(tail_ids.begin(), tail_ids.end(), [&](int a, int b) {
      if (sol.t[a] != sol.t[b]) {
        return sol.t[a] < sol.t[b];
      }
      return a < b;
    });
    for (int id : tail_ids) {
      epilogue_orders.push_back(
          String(std::to_string(epilogue_iter) + "-" + std::to_string(id)));
    }

    annotations.Set("prologue_orders", prologue_orders);
    annotations.Set("body_orders", body_orders);
    annotations.Set("epilogue_orders", epilogue_orders);

    Array<Buffer> used_buffers_array(analysis.used_buffers.begin(),
                                     analysis.used_buffers.end());
    annotations.Set("used_buffers", used_buffers_array);
    annotations.Set("versioned_buffers", Array<Buffer>());

    Stmt body = this->VisitStmt(op->body);
    For new_loop = loop;
    ForNode* loop_ptr = new_loop.CopyOnWrite();
    loop_ptr->body = body;
    loop_ptr->annotations = annotations;
    return new_loop;
  }

  PrimFunc func_;
  ASTTraverser traverser_;
  bool debug_{false};
};

tvm::transform::Pass SunmmioPipelinePlanningILP(bool debug = false) {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule& m, PassContext ctx) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = SunmmioPipelinePlannerILP::Substitute(f, debug);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.SunmmioPipelinePlanningILP", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SunmmioPipelinePlanningILP",
                        SunmmioPipelinePlanningILP);
}

}  // namespace tl
}  // namespace tvm
