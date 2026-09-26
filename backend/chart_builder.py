"""
backend/chart_builder.py

Generates and validates deterministic native Chart Specifications from controlled tool results.
Supported Chart Types: bar, line, scatter, histogram, pie, heatmap
"""

import math
from typing import Dict, List, Any, Optional

ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "pie", "heatmap"}

def validate_chart_spec(spec: Any) -> Optional[Dict[str, Any]]:
    """Validates and sanitizes a chart specification object.
    
    Returns sanitized chart_spec dict if valid, or None if malformed/invalid.
    Protects against code injection, unknown types, and non-serializable objects.
    """
    if not isinstance(spec, dict):
        return None
    
    chart_type = spec.get("chart_type")
    if not isinstance(chart_type, str) or chart_type.lower() not in ALLOWED_CHART_TYPES:
        return None
    
    chart_type_clean = chart_type.lower()
    title = str(spec.get("title", ""))[:150]
    x_axis = str(spec.get("x", ""))[:100]
    y_axis = str(spec.get("y", ""))[:100]
    
    raw_data = spec.get("data")
    if not isinstance(raw_data, list):
        return None
    
    # Sanitize data list
    clean_data = []
    for item in raw_data:
        if isinstance(item, dict):
            clean_item = {}
            for k, v in item.items():
                if not isinstance(k, str):
                    continue
                k_clean = str(k)[:50]
                if isinstance(v, (int, float)):
                    if math.isnan(v) or math.isinf(v):
                        clean_item[k_clean] = None
                    else:
                        clean_item[k_clean] = round(v, 4) if isinstance(v, float) else v
                elif isinstance(v, (str, bool, type(None))):
                    v_str = str(v)[:200] if isinstance(v, str) else v
                    # Check for script injection attempts in strings
                    if isinstance(v_str, str) and ("<script" in v_str.lower() or "javascript:" in v_str.lower() or "eval(" in v_str.lower()):
                        return None
                    clean_item[k_clean] = v_str
            clean_data.append(clean_item)
    
    options = spec.get("options", {})
    clean_options = {}
    if isinstance(options, dict):
        for k, v in options.items():
            if isinstance(k, str) and isinstance(v, (str, int, float, bool, list)):
                clean_options[k[:50]] = v
    
    return {
        "chart_type": chart_type_clean,
        "title": title,
        "x": x_axis,
        "y": y_axis,
        "data": clean_data,
        "options": clean_options
    }

def generate_chart_spec_from_tool_result(tool_name: str, tool_res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generates a deterministic chart specification from an analytical tool result."""
    if not isinstance(tool_res, dict) or not tool_res.get("success", True):
        return None
    
    if tool_name == "top_n":
        col = tool_res.get("column", "Value")
        rows = tool_res.get("results", [])
        if not rows or not isinstance(rows, list):
            return None
        
        data = []
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                label = row.get("name") or row.get("label") or row.get("category") or f"Row #{idx+1}"
                val = row.get(col)
                try:
                    num_val = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    num_val = 0.0
                data.append({"x": str(label), "y": num_val})
        
        if not data:
            return None
        
        spec = {
            "chart_type": "bar",
            "title": f"Top {len(data)} Rows by {col}",
            "x": "Observation",
            "y": col,
            "data": data,
            "options": {"ascending": tool_res.get("ascending", False)}
        }
        return validate_chart_spec(spec)

    elif tool_name == "group_aggregate":
        gb = tool_res.get("group_by", "Group")
        mc = tool_res.get("metric_column", "Metric")
        agg = tool_res.get("aggregation", "mean")
        groups = tool_res.get("groups", [])
        if not groups or not isinstance(groups, list):
            return None
        
        data = []
        for g in groups:
            if isinstance(g, dict):
                grp_name = str(g.get("group", ""))
                val = g.get("value")
                if val is not None:
                    try:
                        num_val = float(val)
                        if not (math.isnan(num_val) or math.isinf(num_val)):
                            data.append({"x": grp_name, "y": num_val})
                    except (ValueError, TypeError):
                        pass
        
        if not data:
            return None
        
        chart_type = "pie" if (agg == "count" and len(data) <= 6) else "bar"
        spec = {
            "chart_type": chart_type,
            "title": f"{agg.title()} of {mc} by {gb}",
            "x": gb,
            "y": f"{agg}({mc})",
            "data": data,
            "options": {"aggregation": agg, "limited": tool_res.get("limited", False)}
        }
        return validate_chart_spec(spec)

    elif tool_name == "pivot_aggregate":
        g1 = tool_res.get("group_by_1", "Row Group")
        g2 = tool_res.get("group_by_2", "Col Group")
        mc = tool_res.get("metric_column", "Value")
        agg = tool_res.get("aggregation", "mean")
        rows = tool_res.get("rows", [])
        cols = tool_res.get("columns", [])
        cells = tool_res.get("cells", [])
        
        if not rows or not cols or not cells or not isinstance(cells, list):
            return None
        
        data = []
        for cell in cells:
            if isinstance(cell, dict):
                r_val = cell.get("row")
                c_val = cell.get("col")
                v_val = cell.get("value")
                if r_val is not None and c_val is not None:
                    try:
                        num_val = float(v_val) if v_val is not None else None
                        if num_val is not None and (math.isnan(num_val) or math.isinf(num_val)):
                            num_val = None
                    except (ValueError, TypeError):
                        num_val = None
                    data.append({"row": str(r_val), "col": str(c_val), "value": num_val})
        
        if not data:
            return None
        
        spec = {
            "chart_type": "heatmap",
            "title": f"Heatmap: {agg.title()} of {mc} ({g1} x {g2})",
            "x": g2,
            "y": g1,
            "data": data,
            "options": {
                "rows": [str(r) for r in rows],
                "columns": [str(c) for c in cols],
                "aggregation": agg
            }
        }
        return validate_chart_spec(spec)

    elif tool_name == "column_correlation":
        col_a = tool_res.get("column_a", "Column A")
        col_b = tool_res.get("column_b", "Column B")
        corr = tool_res.get("correlation")
        scatter_points = tool_res.get("sample_points") or tool_res.get("scatter_points")
        
        if scatter_points and isinstance(scatter_points, list) and len(scatter_points) >= 2:
            data = []
            for pt in scatter_points:
                if isinstance(pt, dict) and "x" in pt and "y" in pt:
                    try:
                        x_val = float(pt["x"])
                        y_val = float(pt["y"])
                        if not (math.isnan(x_val) or math.isinf(x_val) or math.isnan(y_val) or math.isinf(y_val)):
                            data.append({"x": x_val, "y": y_val})
                    except (ValueError, TypeError):
                        pass
            if data:
                spec = {
                    "chart_type": "scatter",
                    "title": f"Scatter Plot: {col_a} vs {col_b} (r = {corr})",
                    "x": col_a,
                    "y": col_b,
                    "data": data,
                    "options": {"correlation": corr}
                }
                return validate_chart_spec(spec)
        
        return None

    elif tool_name == "target_association":
        t_col = tool_res.get("target_column", "Target")
        feats = tool_res.get("features", [])
        if not feats or not isinstance(feats, list):
            return None
        
        data = []
        for f in feats:
            if isinstance(f, dict):
                feat_name = f.get("feature")
                score = f.get("score")
                if feat_name and score is not None:
                    try:
                        num_score = float(score)
                        if not (math.isnan(num_score) or math.isinf(num_score)):
                            data.append({"x": str(feat_name), "y": num_score, "method": f.get("method", "")})
                    except (ValueError, TypeError):
                        pass
        
        if not data:
            return None
        
        spec = {
            "chart_type": "bar",
            "title": f"Feature Associations with Target '{t_col}'",
            "x": "Feature",
            "y": "Association Score",
            "data": data,
            "options": {"target_column": t_col, "target_type": tool_res.get("target_type", "")}
        }
        return validate_chart_spec(spec)

    elif tool_name == "column_summary":
        s = tool_res.get("summary", {})
        bins = s.get("histogram_bins") or s.get("bins")
        if bins and isinstance(bins, list) and len(bins) > 0:
            data = [{"x": str(b.get("bin", idx)), "y": float(b.get("count", 0))} for idx, b in enumerate(bins) if isinstance(b, dict)]
            if data:
                spec = {
                    "chart_type": "histogram",
                    "title": f"Distribution of {tool_res.get('column', 'Column')}",
                    "x": tool_res.get("column", "Value"),
                    "y": "Frequency",
                    "data": data,
                    "options": {}
                }
                return validate_chart_spec(spec)
        return None

    return None

def build_multi_tool_chart_spec(exec_res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Inspects multi-tool execution results and returns chart spec for the most visualizable step."""
    if not isinstance(exec_res, dict) or not exec_res.get("success"):
        return None
    
    step_results = exec_res.get("results", [])
    if not isinstance(step_results, list) or not step_results:
        return None
    
    for step_rec in reversed(step_results):
        if not isinstance(step_rec, dict) or not step_rec.get("success"):
            continue
        tool_name = step_rec.get("tool")
        res = step_rec.get("result", {})
        spec = generate_chart_spec_from_tool_result(tool_name, res)
        if spec:
            return spec
    
    return None
