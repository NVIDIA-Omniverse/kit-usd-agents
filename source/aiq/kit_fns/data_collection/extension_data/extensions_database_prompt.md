# Omniverse Extensions Database: Requirements and Constraints

## Executive Summary

This document outlines the requirements for creating a comprehensive database system for Omniverse Extensions that enables precise LLM access to extension source code and documentation while optimizing for token efficiency and avoiding data duplication.

## Core Requirements

### 1. Comprehensive Extension Access
- **Full Source Code Access**: LLM must be able to read entire extension source code for precise analysis
- **Complete File System Visibility**: Access to any file from any extension at any time
- **Documentation Integration**: Direct access to overview, usage, and other documentation files
- **Real-time File Reading**: Mechanism to read files directly from storage without database duplication

### 2. Version Management

#### Extension Versioning
- **Version-Specific Queries**: Support queries for specific extension versions
- **Kit Version Locking**: For a specific Kit version, lock to corresponding extension versions
- **Cross-Version Analysis**: Compare extension behavior across different versions

#### Documentation Versioning
- **Default to Latest**: Unless explicitly requested, use the most recent documentation
- **Legacy Project Support**: When working on older projects, access corresponding documentation versions
- **Evolution Tracking**: Track how documentation changes over time for each extension

### 3. Data Organization Strategy

#### Repository Structure
- **Extension Repository Dataset**: Collect all extensions for each Kit version
- **Incremental Processing**: Only process new or changed extensions
- **Version Deduplication**: Identical extension versions across Kit releases share storage
- **Selective Processing**: Skip extensions that are already available and unchanged

#### Index Database
- **Extension Metadata**: Name, versions, key metrics for each extension
- **Flat Database Design**: Simple structure with extension ID, version, and essential information
- **Searchable Descriptions**: Both simple and comprehensive descriptions when available
- **Performance Metrics**: Key performance and functionality indicators

### 4. Storage Architecture

#### Hybrid Approach
- **JSON Index**: High-level metadata, relationships, and file location pointers
- **File System Backend**: Actual extension files, source code, and documentation
- **Direct File Access**: Load documentation and source files directly from storage
- **Zero Duplication**: Avoid storing file contents in multiple locations

#### Pipeline Design
```
Kit Version Release
    ↓
Discover Extensions
    ↓
Check Existing Extensions
    ↓
Process New/Changed Only
    ↓
Create Index Entry
    ↓
Link to File System Storage
```

## Constraints and Considerations

### 1. Token Efficiency
- **Minimize Token Consumption**: Balance comprehensive access with efficient token usage
- **Lazy Loading**: Load complete extension data only when specifically requested
- **Smart Caching**: Cache frequently accessed metadata for performance
- **Selective Data Return**: Return only requested information, not full extension dumps

### 2. Performance Requirements
- **Fast Index Queries**: Sub-second response time for metadata searches
- **Scalable Storage**: Handle hundreds of extensions across multiple Kit versions
- **Efficient File Access**: Quick retrieval of individual extension files
- **Memory Management**: Bounded memory usage for index operations

### 3. Data Integrity
- **Content Hash Verification**: Ensure file integrity using checksums
- **Version Consistency**: Maintain consistency between metadata and actual files
- **Dependency Validation**: Track and validate extension dependencies
- **Orphan Prevention**: Identify and clean up unreferenced files

### 4. Processing Constraints
- **Incremental Updates**: Only reprocess when content actually changes
- **Batch Processing**: Handle multiple extensions efficiently
- **Error Recovery**: Robust handling of processing failures
- **Resource Limits**: Respect system resource constraints during processing

## Functional Specifications

### 1. Extension Discovery
```
Query: "List extensions for Kit version X"
Response: Extension metadata from index (lightweight)

Query: "Find extensions related to UI"
Response: Filtered list with relevance scoring
```

### 2. Version Resolution
```
Query: "Show documentation for extension Y"
Default: Latest version documentation

Query: "Show documentation for extension Y in Kit 105.1.0"
Response: Version-specific documentation for that Kit release
```

### 3. File System Access
```
Query: "Read source file Z from extension Y"
Response: Direct file content from Kit extscache location
Example: extscache/omni.ui-2.27.2/omni/ui/_ui.py

Query: "Show extension configuration"
Response: extension.toml from Kit build location
Example: extscache/omni.ui-2.27.2/config/extension.toml
```

### 4. Documentation Access
```
Query: "Show overview documentation for extension Y"
Response: README.md content from Kit extscache location
Example: extscache/omni.ui-2.27.2/docs/README.md

Query: "Show usage examples for extension Y"
Response: Examples from Kit build structure
Example: extscache/omni.ui-2.27.2/examples/
```

## Architecture Implementation

### Database Components

#### 1. Master Index (JSON)
- Extension registry with basic metadata
- Version mappings between Kit and extension versions
- File system location pointers
- Quick access metrics and descriptions

#### 2. File System Backend (Kit Build Structure)
```
extscache/
├── omni.kit.actions.core-1.0.0/     # Extension with version in folder name
│   ├── omni/                         # Python package structure
│   │   └── kit/
│   │       └── actions/
│   ├── config/                       # Extension configuration
│   │   └── extension.toml
│   ├── docs/                         # Documentation files
│   │   ├── README.md
│   │   └── CHANGELOG.md
│   └── examples/                     # Code examples
├── omni.ui-2.27.2/                   # Another extension with version
│   ├── omni/
│   │   └── ui/
│   ├── config/
│   ├── docs/
│   └── examples/
└── omni.kit.window.extensions-1.2.3/
    ├── omni/
    ├── config/
    ├── docs/
    └── examples/

# Index references these actual Kit build locations
extensions_index.json:
{
  "omni.kit.actions.core": {
    "version": "1.0.0",
    "storage_path": "extscache/omni.kit.actions.core-1.0.0/",
    "config_file": "extscache/omni.kit.actions.core-1.0.0/config/extension.toml",

    "description": "Core actions framework for Kit applications",
    "long_description": "Provides fundamental action management and execution capabilities for Omniverse Kit. Includes command registration, undo/redo functionality, and action dispatching systems that enable interactive application development.",
    "topics": ["actions", "commands", "undo", "redo", "framework", "core"],

    "has_python_api": true,
    "has_overview": true,
    "full_api_details_path": "extensions_apis/omni.kit.actions.core-1.0.0-api.json",
    "embedding_vector_index": 42
  },

  "omni.ui": {
    "version": "2.27.2",
    "storage_path": "extscache/omni.ui-2.27.2/",
    "config_file": "extscache/omni.ui-2.27.2/config/extension.toml",

    "description": "Core UI framework for creating graphical interfaces",
    "long_description": "Comprehensive UI toolkit providing widgets, layouts, styling, and event handling for Omniverse Kit applications. Features modern UI components, responsive layouts, theming support, and integration with rendering pipeline.",
    "topics": ["ui", "widgets", "layout", "styling", "window", "controls", "framework"],

    "has_python_api": true,
    "has_overview": true,
    "full_api_details_path": "extensions_apis/omni.ui-2.27.2-api.json",
    "embedding_vector_index": 125
  },

  "omni.kit.window.extensions": {
    "version": "1.2.3",
    "storage_path": "extscache/omni.kit.window.extensions-1.2.3/",
    "config_file": "extscache/omni.kit.window.extensions-1.2.3/config/extension.toml",

    "description": "Extension management window and interface",
    "long_description": "Provides user interface for browsing, installing, enabling, and configuring Omniverse Kit extensions. Includes extension discovery, dependency management, and configuration panels for extension settings.",
    "topics": ["extensions", "management", "window", "ui", "configuration"],

    "has_python_api": false,
    "has_overview": true,
    "full_api_details_path": null,
    "embedding_vector_index": 278
  }
}
```

### API Details Database Structure

The `full_api_details_path` points to structured JSON files containing complete API documentation using the **exact Code Atlas format** from the LC Agent system. 

**IMPORTANT**: The API format must match the proven structure in `source/modules/lc_agent/src/lc_agent/code_atlas/` for compatibility with existing tools and query capabilities.

See `extensions_database_code_atlas_format.md` for the complete specification and detailed examples based on the actual implementation.

## Processing Pipeline Requirements

The extension API extraction must follow the Code Atlas processing pipeline to ensure compatibility:

### 1. AST-Based Analysis
- Use `ast.parse()` to analyze Python source files in extensions
- Extract docstrings using `ast.get_docstring()`
- Parse type annotations and method signatures
- Track import statements and class usage patterns

### 2. Pydantic Model Usage
- Use existing Code Atlas models: `CodeAtlasModuleInfo`, `CodeAtlasClassInfo`, `CodeAtlasMethodInfo`, `CodeAtlasArgumentInfo`
- Serialize using `model_dump(by_alias=True, exclude_defaults=True)`
- Maintain exact field names and data types

### 3. Extension Detection
- Parse `extension.toml` files to identify extension boundaries  
- Set `extension_name` field in root modules
- Handle equivalent modules through wildcard imports

### 4. Processing Pipeline
1. **Discovery Stage**: Scan Kit version for extensions
2. **Comparison Stage**: Check against existing extensions using content hashes
3. **Processing Stage**: Process only new or changed extensions
4. **Index Update**: Add metadata to searchable index
5. **Validation Stage**: Verify data integrity and accessibility

### API Interface Design

#### Quick Metadata Queries
- Extension search and filtering
- Version compatibility checks
- Dependency relationship queries
- Category and functionality browsing

#### Deep Content Access
- Complete source code reading
- Documentation file loading
- Configuration analysis
- Example code retrieval

## Success Metrics

### Performance Targets
- **Index Query Response**: < 200ms for metadata queries
- **File Access Time**: < 500ms for individual file reads
- **Storage Efficiency**: 60% reduction in duplicate data
- **Processing Speed**: Handle 100+ extensions per hour

### Quality Indicators
- **Coverage Completeness**: 100% of available extensions indexed
- **Data Accuracy**: Zero discrepancies between index and storage
- **Version Consistency**: Perfect mapping between Kit and extension versions
- **Access Reliability**: 99.9% successful file access rate

## Future Considerations

### Extensibility
- Support for additional metadata types
- Integration with external documentation sources
- Custom extension categorization schemes
- Advanced semantic search capabilities

### Scalability
- Support for thousands of extensions
- Distributed storage backend options
- Caching layer for high-frequency queries
- Automated monitoring and maintenance tools

---

*This specification provides the foundation for building a comprehensive, efficient, and maintainable Omniverse Extensions database that serves both automated LLM queries and human development workflows.*