
py_test(
    name = "hybrid_attention_config_binding_test",
    srcs = [
        "hybrid_attention_config_binding_test.py",
    ],
    data = ["//rtp_llm:libth_transformer_config_pyi"],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)

py_test(
    name = "server_config_setup_test",
    srcs = [
        "server_config_setup_test.py",
    ],
    data = [
        "//rtp_llm/test/model_test/fake_test/testdata:testdata",
        "//rtp_llm:sdk"
    ],
    deps = [
        "//rtp_llm:testlib",
    ],
    exec_properties = {'gpu':'H20'},
)
