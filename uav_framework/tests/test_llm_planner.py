from uav_framework.llm.llama_interface import LLMPlanner


def test_llm_planner_zero_uavs_makes_no_request():
    planner = LLMPlanner()
    assert planner.plan([], []) == []
    assert planner.request_count == 0
