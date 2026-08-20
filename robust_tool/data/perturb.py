"""Perturbation labels reserved for the Week 3 robustness benchmark."""

from enum import Enum


class PerturbationKind(str, Enum):
    SIMILAR_TOOL_DISTRACTOR = "similar_tool_distractor"
    TOOL_ORDER_SHUFFLE = "tool_order_shuffle"
    TOOL_DESCRIPTION_REWRITE = "tool_description_rewrite"
    TOOL_NAME_SIMILARITY = "tool_name_similarity"
    MISSING_TOOL = "missing_tool"
    TOOL_FAILURE = "tool_failure"
    NOISY_TOOL_RESPONSE = "noisy_tool_response"
    PARTIAL_TOOL_RESPONSE = "partial_tool_response"
    AMBIGUOUS_USER_QUERY = "ambiguous_user_query"
    IRRELEVANT_TOOL_ADDED = "irrelevant_tool_added"
