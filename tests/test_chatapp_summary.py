from chatapp import _answer_from_summary_text, _is_low_quality_answer


def test_summary_answer_prefers_definition_line():
    summary = """
    Overview: quick summary.
    Key points:
    - Optimizing Web Performance: A Case Study on Apache Server and WordPress Integration.
    - Apache Server is an open-source web server software that is highly customizable and widely used.
    """
    answer = _answer_from_summary_text("What is Apache Server?", summary)
    assert "Apache Server is an open-source web server software" in answer


def test_summary_answer_avoids_contact_noise():
    summary = """
    Key points:
    - ANY QUESTIONS? yourmail@example.com +91 620 421 838 yourwebsite.com
    - Apache Server is an open-source web server software.
    """
    answer = _answer_from_summary_text("What is Apache Server?", summary)
    assert "example.com" not in answer.lower()
    assert "apache server is" in answer.lower()


def test_incomplete_definition_marked_low_quality():
    answer = "WORDPRESS BASICS WordPress is a"
    assert _is_low_quality_answer(answer, "What is WordPress?")
