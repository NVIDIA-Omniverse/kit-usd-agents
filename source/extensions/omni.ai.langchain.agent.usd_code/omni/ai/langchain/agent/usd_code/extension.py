## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os

import carb.settings
import omni.ext
from lc_agent import get_node_factory
from lc_agent_usd import USDCodeGenNetworkNode, USDCodeGenNode
from lc_agent_usd import USDCodeNetworkNode as USDCodeDevNode
from lc_agent_usd import USDKnowledgeNode

from .nodes.scene_info_gen_node import SceneInfoGenNode
from .nodes.usd_code_interactive_network_node import USDCodeInteractiveNetworkNode
from .nodes.usd_code_interactive_node import USDCodeInteractiveNode
from .nodes.usd_code_network_node import USDCodeNetworkNode
from .nodes.usd_code_node import USDCodeNode


def get_api_key():
    import carb.settings

    api_key = None
    settings = carb.settings.get_settings().get("/exts/omni.ai.chat_usd.bundle/nvidia_api_key")
    if settings:
        api_key = settings
    else:
        import os

        api_key = os.environ.get("NVIDIA_API_KEY")

    return api_key


def get_embedding_func_id():
    import carb.settings

    func_id = carb.settings.get_settings().get("/exts/omni.ai.chat_usd.bundle/nvidia_embedding_func_id")
    if not func_id:
        import os

        func_id = os.environ.get("NVIDIA_EMBEDDING_FUNC_ID")

    return func_id


def get_embedding_model():
    import carb.settings

    model = carb.settings.get_settings().get("/exts/omni.ai.chat_usd.bundle/nvidia_embedding_model")

    return model or None


class USDCodeExtension(omni.ext.IExt):
    # List of items to hide from the code interpreter
    CODE_INTERPRETER_HIDE_ITEMS = [
        "os.remove",
        "os.unlink",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.mkdir",
        "os.makedirs",
        "os.removedirs",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copyfileobj",
        "shutil.move",
        "shutil.rmtree",
        "pathlib.Path.unlink",
        "pathlib.Path.rmdir",
        "pathlib.Path.mkdir",
        "pathlib.Path.rename",
        "pathlib.Path.replace",
        "pathlib.Path.touch",
        "tempfile.NamedTemporaryFile",
        "tempfile.TemporaryFile",
        "tempfile.mkstemp",
        "tempfile.mkdtemp",
        "mmap.mmap.write",
        "pickle.dump",  # Assuming we want to disable file-based pickling
        "tarfile.open",  # Write modes will be handled in the function
        "zipfile.ZipFile",  # Write modes will be handled in the function
        "gzip.open",  # Write modes will be handled in the function
        "bz2.open",  # Write modes will be handled in the function
        "lzma.open",  # Write modes will be handled in the function
    ]

    def on_startup(self, ext_id):
        self.enable_code_interpreter = carb.settings.get_settings().get(
            "/exts/omni.ai.langchain.agent.usd_code/enable_code_interpreter"
        )

        enable_interpreter_security = carb.settings.get_settings().get(
            "/exts/omni.ai.langchain.agent.usd_code/enable_interpreter_security"
        )
        if enable_interpreter_security:
            code_interpreter_hide_items = self.CODE_INTERPRETER_HIDE_ITEMS
        else:
            code_interpreter_hide_items = None

        enable_scene_info = carb.settings.get_settings().get("/exts/omni.ai.langchain.agent.usd_code/enable_scene_info")
        self.enable_rag_metafunctions = carb.settings.get_settings().get(
            "/exts/omni.ai.langchain.agent.usd_code/enable_rag_metafunctions"
        )
        self.enable_code_atlas = carb.settings.get_settings().get(
            "/exts/omni.ai.langchain.agent.usd_code/enable_code_atlas"
        )
        enable_interpreter_undo_stack = carb.settings.get_settings().get(
            "/exts/omni.ai.langchain.agent.usd_code/enable_undo_stack"
        )
        max_retries = carb.settings.get_settings().get("/exts/omni.ai.langchain.agent.usd_code/max_retries")

        chat_usd_developer_mode = carb.settings.get_settings().get(
            "/exts/omni.ai.chat_usd.bundle/chat_usd_developer_mode"
        )
        chat_usd_developer_mode = chat_usd_developer_mode or os.environ.get("USD_AGENT_DEV_MODE")

        need_rags = chat_usd_developer_mode

        # currently not available outside of NVIDIA
        if need_rags and self.enable_rag_metafunctions:
            api_key = get_api_key()
            if api_key:
                from lc_agent_retrievers import register_all as register_retrievers

                register_retrievers(
                    top_k=15, api_key=api_key, func_id=get_embedding_func_id(), model=get_embedding_model()
                )

        # USD Code and USD Code Interactive use local rag & systems with the llama3 NIM model
        if self.enable_code_interpreter:
            get_node_factory().register(
                USDCodeInteractiveNetworkNode,
                name="USD Code Interactive",
                scene_info=enable_scene_info,
                enable_code_interpreter=self.enable_code_interpreter,
                code_interpreter_hide_items=code_interpreter_hide_items,
                enable_code_atlas=need_rags and self.enable_code_atlas,
                enable_metafunctions=need_rags and self.enable_rag_metafunctions,
                enable_interpreter_undo_stack=enable_interpreter_undo_stack,
                max_retries=max_retries,
            )
        get_node_factory().register(
            USDCodeNetworkNode,
            name="USD Code",
            enable_code_interpreter=self.enable_code_interpreter,
            code_interpreter_hide_items=code_interpreter_hide_items,
        )

        # only available for nvidia developement
        if chat_usd_developer_mode:
            get_node_factory().register(
                USDCodeDevNode,
                name="USD Code Dev",
                enable_code_interpreter=self.enable_code_interpreter,
                code_interpreter_hide_items=code_interpreter_hide_items,
            )
            get_node_factory().register(USDCodeGenNetworkNode, snippet_verification=True, hidden=True)

        # Hiddens
        get_node_factory().register(USDCodeInteractiveNode, hidden=True)
        get_node_factory().register(USDCodeNode, hidden=True)
        get_node_factory().register(SceneInfoGenNode, hidden=True)

        # Hide from lc_agent_usd
        get_node_factory().register(USDCodeGenNode, hidden=True)
        get_node_factory().register(USDKnowledgeNode, hidden=True)

    def on_shutdown(self):
        if self.enable_code_interpreter:
            get_node_factory().unregister(USDCodeInteractiveNetworkNode)
        get_node_factory().unregister(USDCodeNetworkNode)
        get_node_factory().unregister(USDCodeNode)

        get_node_factory().unregister(USDCodeGenNode)
        get_node_factory().unregister(USDKnowledgeNode)

        get_node_factory().unregister(USDCodeInteractiveNode)
        get_node_factory().unregister(SceneInfoGenNode)

        chat_usd_developer_mode = carb.settings.get_settings().get(
            "/exts/omni.ai.chat_usd.bundle/chat_usd_developer_mode"
        )
        chat_usd_developer_mode = chat_usd_developer_mode or os.environ.get("USD_AGENT_DEV_MODE")

        if chat_usd_developer_mode:
            get_node_factory().unregister(USDCodeDevNode)
            get_node_factory().unregister(USDCodeGenNetworkNode)

        if chat_usd_developer_mode and self.enable_rag_metafunctions:
            from lc_agent_retrievers import unregister_all as unregister_retrievers

            unregister_retrievers()
