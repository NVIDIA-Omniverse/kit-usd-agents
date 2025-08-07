-- Use folder name to build extension name and tag. Version is specified explicitly.
local ext = get_current_extension_info()

-- Link resources and the pip bundle into the extension target directory.
project_ext (ext)
    repo_build.prebuild_link {
        { "data", ext.target_dir.."/data" },
        { "docs", ext.target_dir.."/docs" },
        { "omni", ext.target_dir.."/omni" },
        { "%{root}/_build/target-deps/pip_aiq_prebundle", ext.target_dir.."/pip_aiq_prebundle" },
    }