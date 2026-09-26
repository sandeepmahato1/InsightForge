import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from backend.data_engine import DataEngine
from backend.llm_service import LLMService

class MultiAgentOrchestrator:
    def __init__(self, data_engine: DataEngine):
        self.data_engine = data_engine
        self.llm_service = LLMService()
        self.trace_logs = []

    def _add_trace(self, agent_name: str, step: str, details: str, status: str = "completed"):
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "agent": agent_name,
            "step": step,
            "details": details,
            "status": status
        }
        self.trace_logs.append(entry)
        return entry

    def run_investigation(self, research_question: str) -> Dict[str, Any]:
        """Runs controlled autonomous multi-agent pipeline scoped by analytical intent."""
        self.trace_logs = []

        # 1. Intent Classification via LLM Service
        intent = self.llm_service.classify_intent(research_question)
        
        self._add_trace(
            "Root Orchestrator", 
            "Task Planning & Intent Classification", 
            f"Received query: '{research_question}'. Classified intent: '{intent}'. Allocating optimal agent execution path."
        )

        # 2. Data Agent (Always executed for baseline schema & profile)
        self._add_trace("Data Agent", "Data Ingestion & Type Profiling", "Loading dataset schema, identifying numerical/categorical variables and data types.")
        summary = self.data_engine.get_summary_profile()
        self._add_trace(
            "Data Agent", 
            "Data Profiling Complete", 
            f"Ingested {summary['num_rows']} rows and {summary['num_cols']} columns from '{summary['filename']}'. Missing data: {summary['missing_total']} cells."
        )

        # Initialize optional agent result containers
        problem_type, target_col = self.data_engine.detect_problem_type(research_question)
        eda_results = {}
        hypotheses = []
        data_quality_issues = []
        model_selection_info = {}
        ml_results = {"benchmarks": [], "best_model_name": "", "feature_importance": []}
        feature_importance = []

        # 3. Data Quality Module (ALWAYS RUN for complete investigation)
        self._add_trace("Data Quality Agent", "Data Quality Audit", "Auditing missing values, duplicate rows, constant features, high cardinality, and outlier spikes.")
        data_quality_issues = self.data_engine.check_data_quality(target_col)
        self._add_trace(
            "Data Quality Agent",
            "Quality Audit Complete",
            f"Identified {len(data_quality_issues)} quality findings across dataset columns."
        )

        # 4. Analysis Agent (EDA) — ALWAYS RUN
        self._add_trace("Analysis Agent", "Exploratory Data Analysis", "Computing feature correlations, statistical distributions, and target relationships.")
        eda_results = self.data_engine.perform_eda(target_col)
        self._add_trace(
            "Analysis Agent", 
            "EDA Complete", 
            f"Target variable evaluated as '{target_col}'. Correlation matrix and column distributions computed."
        )

        # 5. Hypothesis Agent — ALWAYS RUN
        self._add_trace("Hypothesis Agent", "Formulating & Ranking Hypotheses", "Generating testable domain hypotheses using statistical evidence matrix.")
        hypotheses = self.data_engine.evaluate_hypotheses(research_question, target_col)
        top_h_title = hypotheses[0]['title'] if hypotheses else "Feature Variance Impact"
        self._add_trace(
            "Hypothesis Agent", 
            "Hypothesis Testing Completed", 
            f"Evaluated {len(hypotheses)} hypotheses against statistical evidence. Top hypothesis: '{top_h_title}'."
        )

        # 6. Model Selection & ML Experiment Agent — ALWAYS RUN
        self._add_trace("Task Classifier Agent", "Problem Category Detection", f"Determined ML Problem Type: '{problem_type}' (Target: '{target_col}').")
        
        self._add_trace("Model Selection Agent", "Candidate Algorithm Strategy", "Analyzing sample scale, feature types, cardinality, and imbalance to filter model candidates.")
        model_selection_info = self.data_engine.select_candidate_models(problem_type, target_col, research_question)
        selected_cands = model_selection_info["selected_candidates"]
        primary_metric = model_selection_info["primary_metric"]
        
        self._add_trace(
            "Model Selection Agent",
            "Candidate Pool Allocated",
            f"Selected {len(selected_cands)} algorithms: {', '.join(selected_cands)}. Evaluation metric: '{primary_metric}'."
        )

        self._add_trace("Experiment Agent", f"ML Suite Execution ({problem_type})", f"Cross-validating {len(selected_cands)} candidate models.")
        ml_results = self.data_engine.train_and_benchmark_models(target_col, problem_type, candidate_list=selected_cands)
        
        # TASK 6: Remove fabricated numerical default (0.95)
        benchmarks_list = ml_results.get('benchmarks', [])
        best_score = benchmarks_list[0].get('primary_score') if benchmarks_list else None
        score_str = f"{best_score}" if best_score is not None else "N/A"
        
        self._add_trace(
            "Experiment Agent", 
            "ML Benchmarking Complete", 
            f"Evaluated models. Winning model: '{ml_results['best_model_name']}' with top score {score_str}."
        )

        # 7. Explainability Agent — ALWAYS RUN
        self._add_trace("Explainability Agent", "SHAP Feature Importance Analysis", "Computing feature attribution scores to interpret model predictions.")
        feature_importance = ml_results["feature_importance"]
        top_features_list = [f["feature"] for f in feature_importance[:3]] if feature_importance else ["primary_feature"]
        
        self._add_trace(
            "Explainability Agent", 
            "Model Interpretation Complete", 
            f"Top predictive drivers: {', '.join(top_features_list)}. Feature importances generated."
        )

        # 8. Answer Agent (Synthesizes AI Data Scientist response based strictly on actual Python results)
        self._add_trace("Answer Agent", "Evidence Synthesis", "Synthesizing deterministic computed evidence into structured natural language response.")
        
        evidence_payload = {
            "summary": summary,
            "eda": eda_results,
            "target_column": target_col,
            "hypotheses": hypotheses,
            "data_quality": data_quality_issues,
            "model_selection": model_selection_info,
            "ml_results": ml_results,
            "feature_importance": feature_importance,
            "top_feature": feature_importance[0]["feature"] if feature_importance else (hypotheses[0]["title"] if hypotheses else "primary_variable")
        }

        ai_answer = self.llm_service.synthesize_answer(intent, research_question, evidence_payload)
        
        self._add_trace("Answer Agent", "Answer Synthesized", "Structured AI Data Scientist answer generated.", "completed")

        # 9. Report Agent
        executive_summary = f"""
### Analysis Summary: {research_question}

**Direct Answer:**
{ai_answer['direct_answer']}

**Key Findings:**
""" + "\n".join([f"- {f}" for f in ai_answer['key_findings']])

        if ai_answer.get('model_results_text'):
            executive_summary += f"\n\n**ML Benchmarks:**\n{ai_answer['model_results_text']}"
        if ai_answer.get('explainability_text'):
            executive_summary += f"\n\n**Explainability:**\n{ai_answer['explainability_text']}"
        if ai_answer.get('recommended_next_steps'):
            executive_summary += "\n\n**Recommended Next Steps:**\n" + "\n".join([f"- {r}" for r in ai_answer['recommended_next_steps']])

        self._add_trace("Report Agent", "Investigation Completed", "Final response ready for dashboard delivery.", "finished")

        return {
            "research_question": research_question,
            "intent": intent,
            "problem_type": problem_type,
            "target_column": target_col,
            "agent_trace": self.trace_logs,
            "summary": summary,
            "eda": eda_results,
            "data_quality": data_quality_issues,
            "hypotheses": hypotheses,
            "model_selection": model_selection_info,
            "ml_benchmarks": ml_results["benchmarks"],
            "best_model_name": ml_results["best_model_name"],
            "feature_importance": feature_importance,
            "ai_answer": ai_answer,
            "executive_summary": executive_summary
        }

    def execute_controlled_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Executes approved Python data science tools on demand for conversational extensions.
        
        Performs strict tool name and parameter validation before DataEngine execution.
        Returns structured results on success or structured error dicts on validation failure.
        """
        supported_tools = {"top_n", "column_summary", "filtered_count", "check_data_quality", "summary_profile", "column_correlation", "group_aggregate", "pivot_aggregate", "target_association"}
        
        # TASK 1: Reject unsupported tool names cleanly
        if tool_name not in supported_tools:
            err_msg = f"Tool '{tool_name}' is not supported. Supported tools: {', '.join(sorted(supported_tools))}."
            self._add_trace(
                "Root Orchestrator",
                f"Controlled Tool Validation Failed ({tool_name})",
                err_msg,
                status="failed"
            )
            return {
                "tool": tool_name,
                "success": False,
                "error": err_msg
            }

        df = self.data_engine.df

        # Helper for normalized case-insensitive column resolution
        def resolve_column(col_input: Any) -> Optional[str]:
            if not col_input:
                return None
            target_str = str(col_input).strip()
            target_clean = target_str.lower().replace(' ', '').replace('_', '')
            for c in df.columns:
                if c.lower() == target_str.lower() or c.lower().replace(' ', '').replace('_', '') == target_clean:
                    return c
            return None

        # TASK 2 & 3: Validate tool-specific parameters and return structured errors
        if tool_name == "group_aggregate":
            agg_raw = params.get("aggregation")
            allowed_aggs = {"mean", "median", "sum", "min", "max", "count"}
            if not agg_raw or str(agg_raw).lower().strip() not in allowed_aggs:
                err_msg = f"Invalid or missing parameter 'aggregation': '{agg_raw}'. Allowed aggregations: {', '.join(sorted(allowed_aggs))}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                return {"tool": "group_aggregate", "success": False, "error": err_msg}

            agg_func = str(agg_raw).lower().strip()

            group_by_raw = params.get("group_by")
            if not group_by_raw or isinstance(group_by_raw, (dict, list)):
                err_msg = "Parameter 'group_by' is required for tool 'group_aggregate'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                return {"tool": "group_aggregate", "success": False, "error": err_msg}

            group_by_matched = resolve_column(group_by_raw)
            if not group_by_matched:
                err_msg = f"Column '{group_by_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                return {"tool": "group_aggregate", "success": False, "error": err_msg}

            metric_col_raw = params.get("metric_column")
            metric_col_matched = None
            if agg_func == "count":
                if metric_col_raw and not isinstance(metric_col_raw, (dict, list)):
                    metric_col_matched = resolve_column(metric_col_raw)
                if not metric_col_matched:
                    metric_col_matched = group_by_matched
            else:
                if not metric_col_raw or isinstance(metric_col_raw, (dict, list)):
                    err_msg = f"Parameter 'metric_column' is required for aggregation '{agg_func}'."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                    return {"tool": "group_aggregate", "success": False, "error": err_msg}

                metric_col_matched = resolve_column(metric_col_raw)
                if not metric_col_matched:
                    err_msg = f"Column '{metric_col_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                    return {"tool": "group_aggregate", "success": False, "error": err_msg}

                if not np.issubdtype(df[metric_col_matched].dtype, np.number):
                    err_msg = f"Column '{metric_col_matched}' is non-numeric (type: {df[metric_col_matched].dtype}). Aggregation '{agg_func}' requires a numeric metric column."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                    return {"tool": "group_aggregate", "success": False, "error": err_msg}

            # Drop NaNs
            if agg_func == "count":
                valid_df = df.dropna(subset=[group_by_matched])
            else:
                valid_df = df.dropna(subset=[group_by_matched, metric_col_matched])

            if len(valid_df) == 0:
                err_msg = f"No valid observations for columns '{group_by_matched}' and '{metric_col_matched}'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                return {"tool": "group_aggregate", "success": False, "error": err_msg}

            # High cardinality check
            unique_groups = valid_df[group_by_matched].nunique()
            if unique_groups > 50:
                err_msg = f"Column '{group_by_matched}' has too many unique categories ({unique_groups}) for grouped analysis (maximum limit is 50)."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                return {"tool": "group_aggregate", "success": False, "error": err_msg}

            # Group computation
            if agg_func == "count":
                grouped = valid_df.groupby(group_by_matched, observed=True)[metric_col_matched].count()
            else:
                grouped = valid_df.groupby(group_by_matched, observed=True)[metric_col_matched].agg(agg_func)

            if grouped.empty:
                err_msg = "Grouped aggregation resulted in an empty result set."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (group_aggregate)", err_msg, status="failed")
                return {"tool": "group_aggregate", "success": False, "error": err_msg}

            # Sorting
            if agg_func in ("min",):
                grouped = grouped.sort_values(ascending=True)
            else:
                grouped = grouped.sort_values(ascending=False)

            # Bounding
            limit_n = 15
            total_groups = len(grouped)
            is_limited = total_groups > limit_n
            display_grouped = grouped.head(limit_n)

            # Formatting & JSON serialization guard
            import math
            group_results = []
            for g_key, val in display_grouped.items():
                key_str = str(g_key)
                try:
                    num_val = float(val) if agg_func != "count" else int(val)
                    if isinstance(num_val, float) and (math.isnan(num_val) or math.isinf(num_val)):
                        num_val = None
                except (ValueError, TypeError):
                    num_val = None
                group_results.append({"group": key_str, "value": num_val})

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (group_aggregate)",
                f"Executing Python tool 'group_aggregate' grouping '{group_by_matched}' by '{metric_col_matched}' ({agg_func}).",
                status="completed"
            )
            return {
                "tool": "group_aggregate",
                "success": True,
                "group_by": group_by_matched,
                "metric_column": metric_col_matched,
                "aggregation": agg_func,
                "total_groups": total_groups,
                "limited": is_limited,
                "groups": group_results
            }

        elif tool_name == "pivot_aggregate":
            agg_raw = params.get("aggregation")
            allowed_aggs = {"mean", "median", "sum", "min", "max", "count"}
            if not agg_raw or str(agg_raw).lower().strip() not in allowed_aggs:
                err_msg = f"Invalid or missing parameter 'aggregation': '{agg_raw}'. Allowed aggregations: {', '.join(sorted(allowed_aggs))}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            agg_func = str(agg_raw).lower().strip()

            g1_raw = params.get("group_by_1") or params.get("row_group")
            if not g1_raw or isinstance(g1_raw, (dict, list)):
                err_msg = "Parameter 'group_by_1' is required for tool 'pivot_aggregate'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            g1_matched = resolve_column(g1_raw)
            if not g1_matched:
                err_msg = f"Column '{g1_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            g2_raw = params.get("group_by_2") or params.get("column_group")
            if not g2_raw or isinstance(g2_raw, (dict, list)):
                err_msg = "Parameter 'group_by_2' is required for tool 'pivot_aggregate'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            g2_matched = resolve_column(g2_raw)
            if not g2_matched:
                err_msg = f"Column '{g2_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            if g1_matched == g2_matched:
                err_msg = f"Parameters 'group_by_1' and 'group_by_2' must be distinct columns, but both resolved to '{g1_matched}'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            metric_col_raw = params.get("metric_column")
            metric_col_matched = None
            if agg_func == "count":
                if metric_col_raw and not isinstance(metric_col_raw, (dict, list)):
                    metric_col_matched = resolve_column(metric_col_raw)
                if not metric_col_matched:
                    metric_col_matched = g1_matched
            else:
                if not metric_col_raw or isinstance(metric_col_raw, (dict, list)):
                    err_msg = f"Parameter 'metric_column' is required for aggregation '{agg_func}'."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                    return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

                metric_col_matched = resolve_column(metric_col_raw)
                if not metric_col_matched:
                    err_msg = f"Column '{metric_col_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                    return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

                if not np.issubdtype(df[metric_col_matched].dtype, np.number):
                    err_msg = f"Column '{metric_col_matched}' is non-numeric (type: {df[metric_col_matched].dtype}). Aggregation '{agg_func}' requires a numeric metric column."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                    return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            # Drop NaNs
            if agg_func == "count":
                valid_df = df.dropna(subset=[g1_matched, g2_matched])
            else:
                valid_df = df.dropna(subset=[g1_matched, g2_matched, metric_col_matched])

            if len(valid_df) == 0:
                err_msg = f"No valid observations for columns '{g1_matched}', '{g2_matched}', and '{metric_col_matched}'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            # High cardinality protection
            g1_unique = valid_df[g1_matched].nunique()
            g2_unique = valid_df[g2_matched].nunique()
            if g1_unique > 30 or g2_unique > 30 or (g1_unique * g2_unique) > 200:
                err_msg = f"Grouping columns '{g1_matched}' ({g1_unique} categories) and '{g2_matched}' ({g2_unique} categories) produce too many unique combinations for 2D pivot analysis (maximum limit is 30 categories per dimension)."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            # 2D Pivot Table Computation
            try:
                if agg_func == "count":
                    if metric_col_matched and metric_col_matched != g1_matched and metric_col_matched != g2_matched:
                        pivot = pd.pivot_table(valid_df, values=metric_col_matched, index=g1_matched, columns=g2_matched, aggfunc="count", fill_value=0, observed=True)
                    else:
                        pivot = pd.crosstab(valid_df[g1_matched], valid_df[g2_matched])
                elif agg_func == "median":
                    pivot = pd.pivot_table(valid_df, values=metric_col_matched, index=g1_matched, columns=g2_matched, aggfunc=np.median, fill_value=np.nan, observed=True)
                else:
                    pivot = pd.pivot_table(valid_df, values=metric_col_matched, index=g1_matched, columns=g2_matched, aggfunc=agg_func, fill_value=np.nan, observed=True)
            except Exception as pe:
                err_msg = f"Pivot table calculation failed: {str(pe)}"
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            if pivot.empty:
                err_msg = "Pivot table aggregation resulted in an empty result set."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (pivot_aggregate)", err_msg, status="failed")
                return {"tool": "pivot_aggregate", "success": False, "error": err_msg}

            # Bounding result size
            row_limit = 10
            col_limit = 10
            total_rows = pivot.shape[0]
            total_cols = pivot.shape[1]
            is_limited = (total_rows > row_limit) or (total_cols > col_limit)
            bounded_pivot = pivot.iloc[:row_limit, :col_limit]

            # Formatting & JSON-safe cell extraction
            import math
            row_labels = [str(r) for r in bounded_pivot.index]
            col_labels = [str(c) for c in bounded_pivot.columns]
            cells = []

            for r_idx, r_val in enumerate(bounded_pivot.index):
                r_str = str(r_val)
                for c_idx, c_val in enumerate(bounded_pivot.columns):
                    c_str = str(c_val)
                    raw_val = bounded_pivot.iloc[r_idx, c_idx]
                    try:
                        num_val = float(raw_val) if agg_func != "count" else int(raw_val)
                        if isinstance(num_val, float) and (math.isnan(num_val) or math.isinf(num_val)):
                            num_val = None
                    except (ValueError, TypeError):
                        num_val = None
                    cells.append({"row": r_str, "col": c_str, "value": num_val})

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (pivot_aggregate)",
                f"Executing Python tool 'pivot_aggregate' cross-tabulating '{g1_matched}' x '{g2_matched}' with '{metric_col_matched}' ({agg_func}).",
                status="completed"
            )
            return {
                "tool": "pivot_aggregate",
                "success": True,
                "group_by_1": g1_matched,
                "group_by_2": g2_matched,
                "metric_column": metric_col_matched,
                "aggregation": agg_func,
                "total_rows": total_rows,
                "total_cols": total_cols,
                "limited": is_limited,
                "rows": row_labels,
                "columns": col_labels,
                "cells": cells
            }

        elif tool_name == "column_correlation":
            col_a_raw = params.get("column_a")
            if not col_a_raw:
                err_msg = "Parameter 'column_a' is required for tool 'column_correlation'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            col_b_raw = params.get("column_b")
            if not col_b_raw:
                err_msg = "Parameter 'column_b' is required for tool 'column_correlation'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            col_a_matched = resolve_column(col_a_raw)
            if not col_a_matched:
                err_msg = f"Column '{col_a_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            col_b_matched = resolve_column(col_b_raw)
            if not col_b_matched:
                err_msg = f"Column '{col_b_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            if not np.issubdtype(df[col_a_matched].dtype, np.number):
                err_msg = f"Column '{col_a_matched}' is non-numeric (type: {df[col_a_matched].dtype}). 'column_correlation' requires numeric columns."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            if not np.issubdtype(df[col_b_matched].dtype, np.number):
                err_msg = f"Column '{col_b_matched}' is non-numeric (type: {df[col_b_matched].dtype}). 'column_correlation' requires numeric columns."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            # Check same column request first
            if col_a_matched == col_b_matched:
                valid_s = df[col_a_matched].dropna()
                if len(valid_s) < 2:
                    err_msg = f"Insufficient valid observations for column '{col_a_matched}' (count: {len(valid_s)})."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                    return {"tool": "column_correlation", "success": False, "error": err_msg}
                if float(valid_s.std()) == 0:
                    err_msg = f"Cannot compute correlation for column '{col_a_matched}' with itself because it has zero variance (constant values)."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                    return {"tool": "column_correlation", "success": False, "error": err_msg}

                self._add_trace(
                    "Root Orchestrator",
                    "Controlled Tool Execution (column_correlation - identical columns)",
                    f"Executing Python tool 'column_correlation' on identical column '{col_a_matched}'.",
                    status="completed"
                )
                return {
                    "tool": "column_correlation",
                    "success": True,
                    "column_a": col_a_matched,
                    "column_b": col_b_matched,
                    "correlation": 1.0
                }

            # Check valid pairwise observations for different columns
            s_a = df[col_a_matched]
            s_b = df[col_b_matched]
            valid_df = pd.concat([s_a, s_b], axis=1).dropna()
            if len(valid_df) < 2:
                err_msg = f"Insufficient valid pairwise observations between '{col_a_matched}' and '{col_b_matched}' (count: {len(valid_df)})."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            # Check zero variance / constant columns
            std_a = float(valid_df.iloc[:, 0].std())
            std_b = float(valid_df.iloc[:, 1].std())
            if std_a == 0 or std_b == 0:
                err_msg = f"Cannot compute correlation because one or both columns ('{col_a_matched}', '{col_b_matched}') have zero variance (constant values)."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            # Compute Pearson correlation
            raw_corr = valid_df.iloc[:, 0].corr(valid_df.iloc[:, 1])
            try:
                corr_val = float(raw_corr)
            except (ValueError, TypeError):
                corr_val = float("nan")

            import math
            if math.isnan(corr_val):
                err_msg = f"Correlation calculation for columns '{col_a_matched}' and '{col_b_matched}' resulted in NaN."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_correlation)", err_msg, status="failed")
                return {"tool": "column_correlation", "success": False, "error": err_msg}

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (column_correlation)",
                f"Executing Python tool 'column_correlation' between '{col_a_matched}' and '{col_b_matched}' -> {corr_val:.4f}.",
                status="completed"
            )
            return {
                "tool": "column_correlation",
                "success": True,
                "column_a": col_a_matched,
                "column_b": col_b_matched,
                "correlation": corr_val
            }

        elif tool_name == "top_n":
            col_raw = params.get("column")
            if not col_raw:
                err_msg = "Parameter 'column' is required for tool 'top_n'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (top_n)", err_msg, status="failed")
                return {"tool": "top_n", "success": False, "error": err_msg}

            col_matched = resolve_column(col_raw)
            if not col_matched:
                err_msg = f"Column '{col_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (top_n)", err_msg, status="failed")
                return {"tool": "top_n", "success": False, "error": err_msg}

            # Check numeric data type requirement
            if not np.issubdtype(df[col_matched].dtype, np.number):
                err_msg = f"Column '{col_matched}' is non-numeric (type: {df[col_matched].dtype}). 'top_n' requires a numeric column."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (top_n)", err_msg, status="failed")
                return {"tool": "top_n", "success": False, "error": err_msg}

            # Validate n
            n_raw = params.get("n", 5)
            try:
                n = int(n_raw)
                if n <= 0:
                    raise ValueError("n must be greater than zero.")
                if n > 100:
                    n = 100
            except (ValueError, TypeError):
                err_msg = f"Invalid value for parameter 'n': '{n_raw}'. 'n' must be a positive integer."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (top_n)", err_msg, status="failed")
                return {"tool": "top_n", "success": False, "error": err_msg}

            asc_raw = params.get("ascending", False)
            ascending = True if str(asc_raw).lower() in ("true", "1") else False

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (top_n)",
                f"Executing Python tool 'top_n' on column '{col_matched}' with n={n}, ascending={ascending}.",
                status="completed"
            )
            res = self.data_engine.query_top_n(col_matched, n=n, ascending=ascending)
            return {"tool": "top_n", "success": True, "column": col_matched, "n": n, "ascending": ascending, "results": res}

        elif tool_name == "column_summary":
            col_raw = params.get("column")
            if not col_raw:
                err_msg = "Parameter 'column' is required for tool 'column_summary'."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_summary)", err_msg, status="failed")
                return {"tool": "column_summary", "success": False, "error": err_msg}

            col_matched = resolve_column(col_raw)
            if not col_matched:
                err_msg = f"Column '{col_raw}' does not exist in dataset. Available columns: {list(df.columns)}."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (column_summary)", err_msg, status="failed")
                return {"tool": "column_summary", "success": False, "error": err_msg}

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (column_summary)",
                f"Executing Python tool 'column_summary' on column '{col_matched}'.",
                status="completed"
            )
            res = self.data_engine.query_column_summary(col_matched)
            return {"tool": "column_summary", "success": True, "column": col_matched, "summary": res}

        elif tool_name == "filtered_count":
            # Normalize legacy params ("column", "value") vs structured params ("conditions", "logic")
            conditions = params.get("conditions")
            logic_raw = params.get("logic", "AND")

            if not conditions:
                col_raw = params.get("column")
                val_raw = params.get("value")
                if not col_raw:
                    err_msg = "Parameter 'conditions' (or 'column' and 'value') is required for tool 'filtered_count'."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (filtered_count)", err_msg, status="failed")
                    return {"tool": "filtered_count", "success": False, "error": err_msg}
                if val_raw is None or (isinstance(val_raw, str) and val_raw.strip() == ""):
                    err_msg = "Parameter 'value' is required for tool 'filtered_count'."
                    self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (filtered_count)", err_msg, status="failed")
                    return {"tool": "filtered_count", "success": False, "error": err_msg}
                conditions = [{"column": col_raw, "operator": "==", "value": val_raw}]

            if not isinstance(conditions, list) or len(conditions) == 0:
                err_msg = "Parameter 'conditions' must be a non-empty list."
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (filtered_count)", err_msg, status="failed")
                return {"tool": "filtered_count", "success": False, "error": err_msg}

            res = self.data_engine.query_structured_filtered_count(conditions, logic=str(logic_raw))
            if "error" in res:
                err_msg = res["error"]
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (filtered_count)", err_msg, status="failed")
                return {"tool": "filtered_count", "success": False, "error": err_msg}

            validated_conds = res.get("conditions", [])
            valid_logic = res.get("logic", "AND")

            if len(validated_conds) == 1:
                col_summary = validated_conds[0]["column"]
                val_summary = validated_conds[0]["value"] if validated_conds[0]["operator"] == "==" else f"{validated_conds[0]['operator']} {validated_conds[0]['value']}"
            else:
                col_summary = ", ".join([c["column"] for c in validated_conds])
                val_summary = f"{valid_logic} filter across {len(validated_conds)} conditions"

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (filtered_count)",
                f"Executing Python tool 'filtered_count' with {len(validated_conds)} conditions ({valid_logic}).",
                status="completed"
            )
            return {
                "tool": "filtered_count",
                "success": True,
                "column": col_summary,
                "value": val_summary,
                "metrics": {
                    "matching_count": res["matching_count"],
                    "total_count": res["total_count"],
                    "percentage": res["percentage"]
                },
                "conditions": validated_conds,
                "logic": valid_logic
            }

        elif tool_name == "check_data_quality":
            col_raw = params.get("target_column")
            col_matched = resolve_column(col_raw) if col_raw else None

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (check_data_quality)",
                f"Executing Python tool 'check_data_quality' (target_column: {col_matched}).",
                status="completed"
            )
            res = self.data_engine.check_data_quality(col_matched)
            return {"tool": "check_data_quality", "success": True, "results": res}

        elif tool_name == "summary_profile":
            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (summary_profile)",
                "Executing Python tool 'summary_profile'.",
                status="completed"
            )
            res = self.data_engine.get_summary_profile()
            return {"tool": "summary_profile", "success": True, "results": res}

        elif tool_name == "target_association":
            target_raw = params.get("target_column") or params.get("target")
            target_matched = resolve_column(target_raw) if target_raw else None

            top_n_raw = params.get("n", params.get("top_n", 10))
            try:
                top_n = int(top_n_raw)
            except (ValueError, TypeError):
                top_n = 10

            self._add_trace(
                "Root Orchestrator",
                "Controlled Tool Execution (target_association)",
                f"Executing Python tool 'target_association' (target_column: {target_matched or 'auto-detected'}).",
                status="completed"
            )
            res = self.data_engine.query_target_association(target_col=target_matched, top_n=top_n)
            if "error" in res:
                err_msg = res["error"]
                self._add_trace("Root Orchestrator", "Controlled Tool Validation Failed (target_association)", err_msg, status="failed")
                return {"tool": "target_association", "success": False, "error": err_msg}

            return {
                "tool": "target_association",
                "success": True,
                **res
            }

    def execute_multi_tool_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a multi-tool plan (maximum 2 steps) using internal FilterContext and controlled tools.
        
        Returns structured execution results.
        """
        validation = self.validate_multi_tool_plan(plan)
        if not validation.get("valid"):
            err_msg = validation.get("error", "Plan validation failed.")
            self._add_trace("Root Orchestrator", "Multi-Tool Plan Execution Rejected", err_msg, status="failed")
            return {
                "success": False,
                "route": "MULTI_TOOL_QUERY",
                "error": err_msg,
                "results": []
            }

        validated_plan = validation["plan"]
        steps = validated_plan["steps"]
        step_results = []
        filter_context = None

        for step in steps:
            step_id = step["step_id"]
            tool_name = step["tool"]
            params = step["params"]
            depends_on = step.get("depends_on_step")

            if depends_on is not None:
                prev_step_res = next((r for r in step_results if r["step_id"] == depends_on), None)
                if not prev_step_res or not prev_step_res.get("success"):
                    err_msg = f"Step {step_id} cannot execute because dependency step {depends_on} failed or was not executed."
                    self._add_trace("Root Orchestrator", f"Multi-Tool Step {step_id} Halted", err_msg, status="failed")
                    return {
                        "success": False,
                        "route": "MULTI_TOOL_QUERY",
                        "plan": validated_plan,
                        "error": err_msg,
                        "results": step_results
                    }

                if filter_context and "conditions" in filter_context:
                    conds = filter_context["conditions"]
                    logic = filter_context.get("logic", "AND")
                    mask, val_conds, clean_logic, err = self.data_engine.get_filter_mask(conds, logic)
                    if err:
                        err_msg = f"Failed to reconstruct FilterContext mask for step {step_id}: {err}"
                        self._add_trace("Root Orchestrator", "FilterContext Reconstruction Error", err_msg, status="failed")
                        return {
                            "success": False,
                            "route": "MULTI_TOOL_QUERY",
                            "plan": validated_plan,
                            "error": err_msg,
                            "results": step_results
                        }
                    filtered_df = self.data_engine.df[mask]
                    temp_engine = DataEngine(self.data_engine.file_path, df=filtered_df)
                    temp_orchestrator = MultiAgentOrchestrator(temp_engine)
                    res = temp_orchestrator.execute_controlled_tool(tool_name, params)
                else:
                    res = self.execute_controlled_tool(tool_name, params)
            else:
                res = self.execute_controlled_tool(tool_name, params)

            step_record = {
                "step_id": step_id,
                "tool": tool_name,
                "params": params,
                "depends_on_step": depends_on,
                "success": res.get("success", False),
                "result": res
            }

            if tool_name == "filtered_count" and res.get("success"):
                filter_context = {
                    "conditions": res.get("conditions", []),
                    "logic": res.get("logic", "AND"),
                    "matching_count": res.get("metrics", {}).get("matching_count", 0)
                }
                step_record["filter_context"] = filter_context

            step_results.append(step_record)

            if not res.get("success"):
                err_msg = f"Step {step_id} ({tool_name}) failed: {res.get('error', 'Tool execution failed')}"
                self._add_trace("Root Orchestrator", f"Multi-Tool Step {step_id} Execution Failed", err_msg, status="failed")
                return {
                    "success": False,
                    "route": "MULTI_TOOL_QUERY",
                    "plan": validated_plan,
                    "error": err_msg,
                    "results": step_results
                }

        self._add_trace(
            "Root Orchestrator",
            "Multi-Tool Plan Execution Complete",
            f"Successfully executed {len(step_results)} steps for multi-tool query.",
            status="completed"
        )
        return {
            "success": True,
            "route": "MULTI_TOOL_QUERY",
            "plan": validated_plan,
            "results": step_results,
            "execution_count": len(step_results)
        }

    def validate_multi_tool_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validates a multi-tool execution plan against structure, whitelists, limits, and dependencies.
        
        Returns {"valid": True, "plan": validated_plan} on success,
        or {"valid": False, "error": err_msg} on validation failure.
        """
        if not isinstance(plan, dict):
            return {"valid": False, "error": "Multi-tool plan must be a structured JSON object."}

        route = plan.get("route")
        if route != "MULTI_TOOL_QUERY":
            return {"valid": False, "error": f"Invalid route for multi-tool planner: '{route}'. Expected 'MULTI_TOOL_QUERY'."}

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return {"valid": False, "error": "Parameter 'steps' must be a list of step objects."}

        if len(steps) == 0:
            return {"valid": False, "error": "Multi-tool plan must contain at least 1 step."}

        max_steps = 2
        if len(steps) > max_steps:
            return {"valid": False, "error": f"Multi-tool plan exceeds maximum allowed steps (limit is {max_steps}, got {len(steps)})."}

        supported_tools = {"top_n", "column_summary", "filtered_count", "check_data_quality", "summary_profile", "column_correlation", "group_aggregate", "pivot_aggregate", "target_association"}
        seen_step_ids = set()
        validated_steps = []

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                return {"valid": False, "error": f"Step at index {idx} must be a JSON object."}

            step_id = step.get("step_id")
            if not isinstance(step_id, int) or step_id <= 0:
                return {"valid": False, "error": f"Step at index {idx} missing valid positive integer 'step_id'."}

            if step_id in seen_step_ids:
                return {"valid": False, "error": f"Duplicate step_id {step_id} found in multi-tool plan."}
            seen_step_ids.add(step_id)

            tool_name = step.get("tool")
            if not tool_name or not isinstance(tool_name, str):
                return {"valid": False, "error": f"Step {step_id} missing valid string parameter 'tool'."}

            if tool_name not in supported_tools:
                return {"valid": False, "error": f"Tool '{tool_name}' in step {step_id} is not in the controlled tool whitelist."}

            params = step.get("params")
            if params is None or not isinstance(params, dict):
                return {"valid": False, "error": f"Step {step_id} missing valid dictionary 'params'."}

            # Dependency validation
            depends_on = step.get("depends_on_step")
            if depends_on is not None:
                if not isinstance(depends_on, int):
                    return {"valid": False, "error": f"Step {step_id} parameter 'depends_on_step' must be null or an integer."}

                if depends_on == step_id:
                    return {"valid": False, "error": f"Step {step_id} cannot depend on itself (circular dependency)."}

                if depends_on >= step_id or depends_on not in seen_step_ids:
                    return {"valid": False, "error": f"Step {step_id} references invalid or future dependency step_id {depends_on}."}

                prev_step = [s for s in validated_steps if s["step_id"] == depends_on][0]
                prev_tool = prev_step["tool"]

                valid_dep_matrix = {
                    "filtered_count": {"group_aggregate", "top_n", "column_summary", "column_correlation", "pivot_aggregate", "target_association"},
                    "summary_profile": {"group_aggregate", "column_correlation", "column_summary", "pivot_aggregate", "target_association"},
                    "column_summary": {"column_correlation", "group_aggregate", "pivot_aggregate", "target_association"},
                    "column_correlation": {"group_aggregate", "top_n", "pivot_aggregate", "target_association"},
                    "group_aggregate": {"top_n", "pivot_aggregate", "target_association"},
                    "top_n": {"column_summary", "target_association"}
                }
                allowed_next = valid_dep_matrix.get(prev_tool, set())
                if tool_name not in allowed_next:
                    return {"valid": False, "error": f"Unsupported dependency pair: tool '{tool_name}' in step {step_id} cannot depend on prior tool '{prev_tool}'."}

            # Validate tool parameters using schema check
            param_valid, param_err = self._validate_step_params(tool_name, params)
            if not param_valid:
                return {"valid": False, "error": f"Parameter validation failed for step {step_id} ({tool_name}): {param_err}"}

            validated_steps.append({
                "step_id": step_id,
                "tool": tool_name,
                "params": params,
                "depends_on_step": depends_on
            })

        for i, s in enumerate(validated_steps):
            if s["step_id"] != i + 1:
                return {"valid": False, "error": f"Steps must be numbered sequentially starting from 1 (expected step_id {i+1}, got {s['step_id']})."}

        validated_plan = {
            "route": "MULTI_TOOL_QUERY",
            "steps": validated_steps
        }
        return {"valid": True, "plan": validated_plan}

    def _validate_step_params(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Helper to validate tool parameters without executing or mutating the dataset."""
        df = self.data_engine.df

        def resolve_col(c_in: Any) -> Optional[str]:
            if not c_in:
                return None
            t_str = str(c_in).strip()
            t_clean = t_str.lower().replace(' ', '').replace('_', '')
            for c in df.columns:
                if c.lower() == t_str.lower() or c.lower().replace(' ', '').replace('_', '') == t_clean:
                    return c
            return None

        if tool_name == "summary_profile":
            return True, ""

        elif tool_name == "check_data_quality":
            t_col = params.get("target_column")
            if t_col and not resolve_col(t_col):
                return False, f"Target column '{t_col}' does not exist in dataset."
            return True, ""

        elif tool_name == "column_summary":
            col = params.get("column")
            if not col:
                return False, "Missing parameter 'column'."
            if not resolve_col(col):
                return False, f"Column '{col}' does not exist in dataset."
            return True, ""

        elif tool_name == "top_n":
            col = params.get("column")
            if not col:
                return False, "Missing parameter 'column'."
            col_m = resolve_col(col)
            if not col_m:
                return False, f"Column '{col}' does not exist in dataset."
            if not np.issubdtype(df[col_m].dtype, np.number):
                return False, f"Column '{col_m}' is non-numeric, required for 'top_n'."
            n_raw = params.get("n", 5)
            try:
                n_val = int(n_raw)
                if n_val <= 0:
                    return False, "Parameter 'n' must be positive."
            except (ValueError, TypeError):
                return False, f"Invalid value for 'n': '{n_raw}'."
            return True, ""

        elif tool_name == "column_correlation":
            col_a = params.get("column_a")
            col_b = params.get("column_b")
            if not col_a or not col_b:
                return False, "Parameters 'column_a' and 'column_b' are required."
            a_m = resolve_col(col_a)
            b_m = resolve_col(col_b)
            if not a_m:
                return False, f"Column '{col_a}' does not exist in dataset."
            if not b_m:
                return False, f"Column '{col_b}' does not exist in dataset."
            if not np.issubdtype(df[a_m].dtype, np.number):
                return False, f"Column '{a_m}' is non-numeric, required for 'column_correlation'."
            if not np.issubdtype(df[b_m].dtype, np.number):
                return False, f"Column '{b_m}' is non-numeric, required for 'column_correlation'."
            return True, ""

        elif tool_name == "group_aggregate":
            gb = params.get("group_by")
            agg = params.get("aggregation")
            if not gb:
                return False, "Missing parameter 'group_by'."
            gb_m = resolve_col(gb)
            if not gb_m:
                return False, f"Column '{gb}' does not exist in dataset."
            allowed_aggs = {"mean", "median", "sum", "min", "max", "count"}
            if not agg or str(agg).lower().strip() not in allowed_aggs:
                return False, f"Invalid aggregation '{agg}'. Allowed: {', '.join(sorted(allowed_aggs))}."
            agg_f = str(agg).lower().strip()
            mc = params.get("metric_column")
            if agg_f != "count":
                if not mc:
                    return False, f"Parameter 'metric_column' is required for aggregation '{agg_f}'."
                mc_m = resolve_col(mc)
                if not mc_m:
                    return False, f"Column '{mc}' does not exist in dataset."
                if not np.issubdtype(df[mc_m].dtype, np.number):
                    return False, f"Column '{mc_m}' is non-numeric, required for aggregation '{agg_f}'."
            return True, ""

        elif tool_name == "pivot_aggregate":
            g1 = params.get("group_by_1") or params.get("row_group")
            g2 = params.get("group_by_2") or params.get("column_group")
            agg = params.get("aggregation")
            if not g1 or not g2:
                return False, "Parameters 'group_by_1' and 'group_by_2' are required for 'pivot_aggregate'."
            g1_m = resolve_col(g1)
            g2_m = resolve_col(g2)
            if not g1_m:
                return False, f"Column '{g1}' does not exist in dataset."
            if not g2_m:
                return False, f"Column '{g2}' does not exist in dataset."
            if g1_m == g2_m:
                return False, f"Parameters 'group_by_1' and 'group_by_2' must be distinct columns."
            allowed_aggs = {"mean", "median", "sum", "min", "max", "count"}
            if not agg or str(agg).lower().strip() not in allowed_aggs:
                return False, f"Invalid aggregation '{agg}'. Allowed: {', '.join(sorted(allowed_aggs))}."
            agg_f = str(agg).lower().strip()
            mc = params.get("metric_column")
            if agg_f != "count":
                if not mc:
                    return False, f"Parameter 'metric_column' is required for aggregation '{agg_f}'."
                mc_m = resolve_col(mc)
                if not mc_m:
                    return False, f"Column '{mc}' does not exist in dataset."
                if not np.issubdtype(df[mc_m].dtype, np.number):
                    return False, f"Column '{mc_m}' is non-numeric, required for aggregation '{agg_f}'."
            return True, ""

        elif tool_name == "filtered_count":
            conditions = params.get("conditions")
            if not conditions:
                col = params.get("column")
                val = params.get("value")
                if not col:
                    return False, "Missing parameter 'conditions' (or 'column' and 'value')."
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    return False, "Missing parameter 'value'."
                conditions = [{"column": col, "operator": "==", "value": val}]
            if not isinstance(conditions, list) or len(conditions) == 0:
                return False, "Parameter 'conditions' must be a non-empty list."
            allowed_ops = {"==", "!=", ">", "<", ">=", "<=", "contains"}
            for cond in conditions:
                if not isinstance(cond, dict):
                    return False, "Condition must be an object."
                c_raw = cond.get("column")
                op_raw = cond.get("operator", "==")
                v_raw = cond.get("value")
                if not c_raw:
                    return False, "Condition missing 'column'."
                if v_raw is None or (isinstance(v_raw, str) and str(v_raw).strip() == ""):
                    return False, f"Condition for column '{c_raw}' missing 'value'."
                c_m = resolve_col(c_raw)
                if not c_m:
                    return False, f"Column '{c_raw}' does not exist in dataset."
                op_s = str(op_raw).strip()
                if op_s not in allowed_ops:
                    return False, f"Unsupported operator '{op_raw}'."
                if op_s in (">", "<", ">=", "<="):
                    try:
                        float(v_raw)
                    except (ValueError, TypeError):
                        return False, f"Value '{v_raw}' is not numeric, required for operator '{op_s}'."
                    num_s = pd.to_numeric(df[c_m], errors='coerce')
                    if num_s.isnull().all() and not df[c_m].isnull().all():
                        return False, f"Column '{c_m}' is non-numeric, required for operator '{op_s}'."
            return True, ""

        elif tool_name == "target_association":
            t_col = params.get("target_column") or params.get("target")
            if t_col:
                if not resolve_col(t_col):
                    return False, f"Target column '{t_col}' does not exist in dataset."
            else:
                _, detected_t = self.data_engine.detect_problem_type("", None)
                if not detected_t or detected_t not in df.columns:
                    return False, "Missing parameter 'target_column'."
            return True, ""

        return True, ""



