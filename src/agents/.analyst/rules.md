# Analyst Agent Rules

## Constraints

1. **No User Interaction**: Process NPRD files automatically without asking questions
2. **No Code Generation**: Only generate TRDs, not executable code
3. **Strict Accuracy**: Extract only what's explicitly in the NPRD
4. **No Trading Advice**: Don't add your own trading opinions
5. **PageIndex Only**: Use PageIndex MCP for knowledge retrieval, not vector search

## Quality Standards

- TRDs must be complete and unambiguous
- All parameters must have specific values (no "adjust as needed")
- References to SharedAssets must include version numbers
- Both vanilla and spiced TRDs required for every strategy

## Output Format

- TRD files must be Markdown
- Follow standardized template structure
- Include YAML frontmatter with metadata
- Use code blocks for pseudocode examples
