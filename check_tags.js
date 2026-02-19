
import fs from 'fs';

const content = fs.readFileSync('quantmind-ide/src/lib/components/MainContent.svelte', 'utf8');
const lines = content.split('\n');

let stack = [];
let lineNum = 0;

const tagRegex = /<(div|span|h4|button|section|main|article|header|footer|p|ul|li|ol|i|a|svelte:component|MonacoEditor|SettingsView|PaperTradingPanel|MonteCarloVisualization|FileText|Newspaper|Folder|FolderOpen|Clock|Bot|ChevronRight|Plus|Home|TestTube|Play|BarChart2|PieChart|TrendingUp|DollarSign|Loader|AlertTriangle|RefreshCw|Library|X|Edit3|BookOpen|Boxes)(?:\s+[^>]*?)?>|<\/(div|span|h4|button|section|main|article|header|footer|p|ul|li|ol|i|a|svelte:component|MonacoEditor|SettingsView|PaperTradingPanel|MonteCarloVisualization|FileText|Newspaper|Folder|FolderOpen|Clock|Bot|ChevronRight|Plus|Home|TestTube|Play|BarChart2|PieChart|TrendingUp|DollarSign|Loader|AlertTriangle|RefreshCw|Library|X|Edit3|BookOpen|Boxes)>/g;

lines.forEach((line, index) => {
    lineNum = index + 1;
    let match;
    while ((match = tagRegex.exec(line)) !== null) {
        const fullTag = match[0];
        const tagName = match[1] || match[2];
        const isClosing = fullTag.startsWith('</');
        const isSelfClosing = fullTag.endsWith('/>');

        if (isSelfClosing) continue;

        if (isClosing) {
            if (stack.length === 0) {
                console.log(`Error: Unexpected closing tag </${tagName}> at line ${lineNum}`);
            } else {
                const last = stack.pop();
                if (last.name !== tagName) {
                    console.log(`Error: Tag mismatch. Expected </${last.name}>, found </${tagName}> at line ${lineNum} (opened at line ${last.line})`);
                }
            }
        } else {
            stack.push({ name: tagName, line: lineNum });
        }
    }
});

stack.forEach(tag => {
    console.log(`Error: Unclosed tag <${tag.name}> opened at line ${tag.line}`);
});
