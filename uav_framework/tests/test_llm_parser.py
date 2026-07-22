from uav_framework.agents.planner import GroqPlanner


class DummyUAV:
    def __init__(self, uav_id):
        self.uav_id = uav_id
        self.pos = (0.0, 0.0, 10.0)


def test_groq_response_parser_returns_valid_directions():
    planner = GroqPlanner()
    response = '{"movements":[{"uav_id":0,"dx":1,"dy":0,"dz":-1},{"uav_id":1,"dx":0,"dy":1,"dz":0}]}'
    out, candidate_id = planner._parse_directions(response, [DummyUAV(0), DummyUAV(1)])
    assert out == [(1, 0, -1), (0, 1, 0)]
    assert candidate_id == ""
