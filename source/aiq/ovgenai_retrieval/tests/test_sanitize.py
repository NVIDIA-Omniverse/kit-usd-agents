from ovgenai_retrieval.sanitize import sanitize_identifier, sanitize_query


def test_sanitize_query_strips_html_tags():
    # Tags come out; their text content stays (matches legacy behavior).
    assert sanitize_query("<b>hello</b>") == "hello"
    assert sanitize_query("<script>bad()</script>hello") == "bad()hello"


def test_sanitize_query_escapes_entities():
    assert sanitize_query("a & b") == "a &amp; b"


def test_sanitize_query_removes_control_chars():
    assert sanitize_query("hello\x00\x07world") == "helloworld"


def test_sanitize_query_collapses_whitespace():
    assert sanitize_query("  hello\t  world\n") == "hello world"


def test_sanitize_query_empty():
    assert sanitize_query("") == ""
    assert sanitize_query(None) == ""  # type: ignore[arg-type]


def test_sanitize_identifier_strips_traversal():
    assert sanitize_identifier("../../etc/passwd") == "etc/passwd"


def test_sanitize_identifier_strips_backslash():
    assert sanitize_identifier("omni\\ui\\Window") == "omniuiWindow"


def test_sanitize_identifier_truncates():
    assert sanitize_identifier("x" * 300, max_length=50) == "x" * 50


def test_sanitize_identifier_empty_returns_none():
    assert sanitize_identifier("") is None
    assert sanitize_identifier(None) is None  # type: ignore[arg-type]
    assert sanitize_identifier("   ") is None


def test_sanitize_identifier_strips_html():
    assert sanitize_identifier("<b>Cls</b>") == "Cls"
