# Overview: LangChain AIQ Pip Bundle

This extension exposes the AIQ-centric pip modules to Omniverse Kit.  When the extension is enabled, the following third-party packages become importable by any Python script executed inside Kit:

* **aiq_cosmos 0.1.0**  – Core AIQ Cosmos utilities.
* **lc_agent_aiq 0.1.0** – LangChain Agent helpers tailored for AIQ.

The extension does not provide additional functionality; it simply ensures that the wheels are installed and discoverable at runtime.