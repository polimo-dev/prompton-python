from __future__ import annotations

import pytest

from prompton import template
from prompton.errors import MissingVariableError, RenderError, TemplateSyntaxError


def render(source: str, variables=None, engine="liquid") -> str:
    return template.render(source, variables, engine)


class TestOutput:
    def test_renders_variables_without_escaping(self):
        assert render("{{ html }}", {"html": "<b>&</b>"}) == "<b>&</b>"

    def test_a_present_none_renders_as_empty_but_an_absent_key_is_an_error(self):
        assert render("[{{ x }}]", {"x": None}) == "[]"
        with pytest.raises(MissingVariableError) as error:
            render("[{{ x }}]", {})
        assert error.value.variable == "x"

    def test_nested_access_reports_the_dotted_path(self):
        with pytest.raises(MissingVariableError) as error:
            render("{{ user.name }}", {"user": {}})
        assert error.value.variable == "user.name"

    def test_list_and_index_access(self):
        assert render("{{ rows[1].label }}", {"rows": [{"label": "a"}, {"label": "b"}]}) == "b"
        assert render("{{ items[9] }}", {"items": ["a"]}) == ""

    def test_numbers_keep_liquid_notation(self):
        assert render("{{ n }}", {"n": 2.0}) == "2.0"
        assert render("{{ n }}", {"n": 3}) == "3"
        assert render("{{ b }}", {"b": True}) == "true"


class TestControlFlow:
    def test_untaken_branches_are_not_checked(self):
        assert render('{% if mode == "a" %}{{ only_a }}{% endif %}', {"mode": "b"}) == ""

    def test_taken_branches_are(self):
        with pytest.raises(MissingVariableError):
            render("{% if flag %}{{ other }}{% endif %}", {"flag": True})

    def test_unless_reports_a_missing_condition_variable(self):
        with pytest.raises(MissingVariableError) as error:
            render("{% unless flag %}off{% endunless %}", {})
        assert error.value.variable == "flag"

    def test_for_else_break_continue(self):
        assert render("{% for i in xs %}{{ i }}{% else %}none{% endfor %}", {"xs": []}) == "none"
        source = '{% for i in xs %}{% if i == "b" %}{% continue %}{% endif %}{{ i }}{% endfor %}'
        assert render(source, {"xs": ["a", "b", "c"]}) == "ac"
        source = '{% for i in xs %}{% if i == "c" %}{% break %}{% endif %}{{ i }}{% endfor %}'
        assert render(source, {"xs": ["a", "b", "c", "d"]}) == "ab"

    def test_a_blank_block_body_renders_as_nothing(self):
        """Liquid's blank-body rule: whitespace-only bodies disappear."""
        source = "{% for i in xs %}{{ i }}{% unless forloop.last %} {% endunless %}{% endfor %}"
        assert render(source, {"xs": ["a", "b"]}) == "ab"
        source = "{% for i in xs %}{{ i }}{% unless forloop.last %},{% endunless %}{% endfor %}"
        assert render(source, {"xs": ["a", "b"]}) == "a,b"

    def test_nested_loops_break_only_the_inner_one(self):
        source = (
            "{% for a in outer %}{% for b in inner %}{% break %}{{ b }}{% endfor %}{{ a }}"
            "{% endfor %}"
        )
        assert render(source, {"outer": [1, 2], "inner": [9]}) == "12"


class TestFilters:
    def test_size_counts_characters_not_bytes(self):
        assert render("{{ s | size }}", {"s": "한글"}) == "2"

    def test_join_defaults_to_a_single_space(self):
        assert render("{{ xs | join }}", {"xs": ["a", "b"]}) == "a b"

    def test_default_applies_to_blank_values_only(self):
        assert render('{{ x | default: "fb" }}', {"x": ""}) == "fb"
        assert render('{{ x | default: "fb" }}', {"x": "v"}) == "v"

    def test_a_filter_outside_the_whitelist_is_refused(self):
        with pytest.raises(RenderError):
            render("{{ s | upcase }}", {"s": "abc"})


class TestParsing:
    @pytest.mark.parametrize(
        "source",
        [
            '{% include "other" %}',
            "{% capture x %}y{% endcapture %}",
            "{% raw %}{{ a }}{% endraw %}",
            "{% case n %}{% when 1 %}one{% endcase %}",
            "{% comment %}hidden{% endcomment %}",
            "{% if a %}no end",
        ],
    )
    def test_rejected_constructs(self, source):
        with pytest.raises(TemplateSyntaxError):
            render(source, {"a": True, "n": 1})

    def test_the_raw_engine_never_parses(self):
        source = '{% include "x" %} {{ a'
        assert render(source, {}, "raw") == source

    def test_a_parsed_template_is_reusable(self):
        parsed = template.parse("Hi {{ name }}")
        assert parsed.render({"name": "a"}) == "Hi a"
        assert parsed.render({"name": "b"}) == "Hi b"


class TestMessages:
    def test_renders_content_and_passes_other_keys_through(self):
        messages = [{"role": "system", "content": "Hi {{ name }}", "name": "bot"}]
        assert template.render_messages(messages, {"name": "Ada"}) == [
            {"role": "system", "content": "Hi Ada", "name": "bot"}
        ]

    def test_a_missing_variable_in_any_message_fails_the_whole_render(self):
        messages = [{"role": "system", "content": "ok"}, {"role": "user", "content": "{{ x }}"}]
        with pytest.raises(MissingVariableError):
            template.render_messages(messages, {})


class TestLint:
    def test_whitespace_control_is_rejected_but_still_renders(self):
        reasons = [reason.as_dict() for reason in template.lint("{%- if a -%}x{%- endif -%}")]
        assert reasons == [
            {"kind": "whitespace_control", "value": "{%-"},
            {"kind": "whitespace_control", "value": "-%}"},
        ]
        assert render("{%- if a -%} x {%- endif -%}", {"a": True}) == "x"

    def test_an_allowed_template_lints_clean(self):
        assert template.lint("{% if a %}{{ b | size }}{% endif %}") == []
