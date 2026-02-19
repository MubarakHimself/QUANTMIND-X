
import fs from 'fs';

const filePath = process.argv[2] || 'quantmind-ide/src/lib/components/MainContent.svelte';
const content = fs.readFileSync(filePath, 'utf8');
const lines = content.split('\n');

let stack = [];

// Improved regex to handle multi-line tags
const tagRegex = /<(\/?)([a-zA-Z0-9:\-]+)(?:\s+[^>]*?)?(\/?)>/g;

let fullContent = content.replace(/<!--[\s\S]*?-->/g, ''); // Remove comments

let match;
while ((match = tagRegex.exec(fullContent)) !== null) {
    const isClosing = match[1] === '/';
    const tagName = match[2];
    const isSelfClosing = match[3] === '/';

    if (isSelfClosing) continue;

    // Ignore some tags that are often self-closing in Svelte or special
    if (['img', 'br', 'hr', 'input', 'link', 'meta'].includes(tagName.toLowerCase())) continue;

    if (isClosing) {
        if (stack.length === 0) {
            const pos = fullContent.substring(0, match.index).split('\n');
            console.log(`Error: Unexpected closing tag </${tagName}> at line ${pos.length}:${pos[pos.length - 1].length + 1}`);
        } else {
            const last = stack.pop();
            if (last.name !== tagName) {
                const pos = fullContent.substring(0, match.index).split('\n');
                console.log(`Error: Tag mismatch at line ${pos.length}. Expected </${last.name}> (opened at ${last.line}), found </${tagName}>`);
            }
        }
    } else {
        const pos = fullContent.substring(0, match.index).split('\n');
        stack.push({ name: tagName, line: pos.length });
    }
}

while (stack.length > 0) {
    const tag = stack.pop();
    console.log(`Error: Unclosed tag <${tag.name}> opened at line ${tag.line}`);
}
