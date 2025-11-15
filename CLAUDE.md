# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Node.js + TypeScript project using `node-llama-cpp` to run local LLM inference with the LiquidAI LFM2-VL-1.6B vision-language model. The project demonstrates chat sessions with streaming responses, JSON schema validation, and structured output parsing.

## Development Commands

### Initial Setup
```bash
npm install  # Downloads dependencies and pulls model files automatically via postinstall hook
```

The `postinstall` script automatically downloads the model file `hf:LiquidAI/LFM2-VL-1.6B-GGUF` to the `./models` directory.

### Running the Project
```bash
npm start  # Runs src/index.ts directly via vite-node (no build required)
npm run start:build  # Runs the compiled version from dist/
```

### Building
```bash
npm run build  # Compiles TypeScript to dist/ (prebuild script cleans output first)
```

### Linting and Formatting
```bash
npm run lint  # Runs ESLint with all configured rules
npm run format  # Auto-fixes ESLint issues where possible
```

### Cleaning
```bash
npm run clean  # Removes node_modules, dist, tsconfig.tsbuildinfo, and models
```

### Model Management
```bash
npm run models:pull  # Manually re-download model files to ./models
```

## Architecture

### Project Structure
- **`src/index.ts`**: Single entry point containing the complete application logic
- **`models/`**: Downloaded GGUF model files (gitignored, auto-downloaded)
- **`dist/`**: Compiled JavaScript output (gitignored)

### Core Application Flow (src/index.ts)
1. **Model Loading**: Resolves and loads the LFM2-VL-1.6B model from the models directory
2. **Context Creation**: Creates a context with max 8096 tokens
3. **Chat Session**: Demonstrates three types of interactions:
   - Streaming response with segment information (chain of thought)
   - Simple follow-up question using conversation history
   - Structured JSON output using grammar-constrained generation

### Key Dependencies
- **`node-llama-cpp`**: Core library for running LLAMA models locally
  - `getLlama()`: Initializes the LLAMA runtime
  - `resolveModelFile()`: Locates model files in the models directory
  - `LlamaChatSession`: Manages conversation context and history
  - Grammar API: Forces responses to match JSON schemas
- **`chalk`**: Terminal color output for better UX
- **`vite-node`**: Fast TypeScript execution without pre-compilation

### Response Streaming
The codebase demonstrates two streaming patterns:
- **Simple streaming** (`onTextChunk`): Plain text chunks (commented out in example)
- **Segment streaming** (`onResponseChunk`): Includes metadata about reasoning segments (e.g., chain of thought markers)

### Grammar-Constrained Generation
The project shows how to force structured JSON responses:
```typescript
const responseGrammar = await llama.createGrammarForJsonSchema({...schema...});
const response = await session.prompt(question, {grammar: responseGrammar});
const parsed = responseGrammar.parse(response);
```

## TypeScript Configuration

- **Target/Module**: ES2022 with ESM modules
- **Strict Mode**: Enabled with additional strict checks (noImplicitAny, noImplicitReturns, etc.)
- **Output**: Compiled to `dist/` with source maps and type declarations
- **Entry Point**: Only `src/index.ts` is explicitly included

## ESLint Configuration

The project uses a comprehensive ESLint setup with:
- **Stylistic rules**: 4-space indentation, double quotes, semicolons required, no object curly spacing
- **Import ordering**: Enforced with specific group ordering (builtin → external → internal → parent → sibling)
- **File extensions**: Must use `.js` extension in imports (n/file-extension-in-import)
- **TypeScript-specific**: Member ordering (fields → constructor → methods), explicit accessibility modifiers
- **JSDoc**: Recommended rules with some relaxation (descriptions not required)

**Ignored directories**: `dist/`, `models/`

## Code Style Notes

- **Indentation**: 4 spaces (not tabs)
- **Quotes**: Double quotes preferred
- **Object spacing**: No spaces inside braces `{like: this}`
- **Brace style**: 1tbs (opening brace on same line)
- **Max line length**: 140 characters
- **Member accessibility**: Must explicitly declare `public`/`private`/`protected` on class members
- **Type literal delimiters**: Use commas, not semicolons (e.g., `{foo: string, bar: number}`)

## Important Notes

- Model files are large (>2GB) and downloaded automatically on `npm install`
- The project requires Node.js >=20.0.0
- ESM modules are required (`"type": "module"` in package.json)
- Development uses `vite-node` for fast iteration without build step
