import { useState, useEffect } from 'react';
import { readDir, DirEntry } from '@tauri-apps/plugin-fs';
import { Folder, FileCode, File, ChevronRight, ChevronDown } from 'lucide-react';
import clsx from 'clsx';
// import { join } from '@tauri-apps/api/path'; // Requires @tauri-apps/api

// Recursive Tree Item
const FileTreeItem = ({ entry, path, onSelect }: { entry: DirEntry, path: string, onSelect: (path: string, isDir: boolean) => void }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [children, setChildren] = useState<DirEntry[]>([]);
    const [loading, setLoading] = useState(false);

    const fullPath = `${path}/${entry.name}`; // Naive join, prefer API join if async

    const toggleOpen = async (e: React.MouseEvent) => {
        e.stopPropagation();
        if (entry.isDirectory) {
            if (!isOpen && children.length === 0) {
                setLoading(true);
                try {
                    // For now using absolute paths if 'path' provides it
                    const entries = await readDir(fullPath);
                    // Sort: Directories first, then files
                    entries.sort((a, b) => {
                        if (a.isDirectory === b.isDirectory) return a.name.localeCompare(b.name);
                        return a.isDirectory ? -1 : 1;
                    });
                    setChildren(entries);
                } catch (err) {
                    console.error("Failed to read dir:", fullPath, err);
                }
                setLoading(false);
            }
            setIsOpen(!isOpen);
        } else {
            onSelect(fullPath, false);
        }
    };

    const Icon = entry.isDirectory ? (isOpen ? ChevronDown : ChevronRight) : FileCode;
    const TypeIcon = entry.isDirectory ? Folder : (entry.name.endsWith('.py') || entry.name.endsWith('.mq5') ? FileCode : File);

    return (
        <div className="pl-2 select-none">
            <div
                className={clsx(
                    "flex items-center gap-1 py-1 px-2 cursor-pointer hover:bg-white/5 rounded text-sm transition-colors",
                    !entry.isDirectory && "text-text-muted hover:text-primary"
                )}
                onClick={toggleOpen}
            >
                <div className="w-4 h-4 flex items-center justify-center opacity-50">
                    {entry.isDirectory && <Icon size={14} />}
                </div>
                <TypeIcon size={14} className={clsx("mr-1", entry.isDirectory ? "text-secondary" : "text-blue-400")} />
                <span className="truncate">{entry.name}</span>
            </div>

            {isOpen && (
                <div className="border-l border-white/5 ml-3">
                    {loading ? (
                        <div className="pl-4 text-xs text-text-muted animate-pulse">Loading...</div>
                    ) : (
                        children.map((child) => (
                            <FileTreeItem key={child.name} entry={child} path={fullPath} onSelect={onSelect} />
                        ))
                    )}
                </div>
            )}
        </div>
    );
};

export default function FileTree({ rootPath, onFileSelect }: { rootPath: string, onFileSelect: (path: string) => void }) {
    const [rootEntries, setRootEntries] = useState<DirEntry[]>([]);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const loadRoot = async () => {
            try {
                console.log("Reading root:", rootPath);
                const entries = await readDir(rootPath);
                entries.sort((a, b) => {
                    if (a.isDirectory === b.isDirectory) return a.name.localeCompare(b.name);
                    return a.isDirectory ? -1 : 1;
                });
                setRootEntries(entries);
            } catch (err) {
                console.error(err);
                setError(`Failed to read: ${rootPath}. Ensure permissions.`);
            }
        };
        loadRoot();
    }, [rootPath]);

    if (error) return <div className="p-4 text-red-500 text-xs">{error}</div>;

    return (
        <div className="flex flex-col h-full overflow-y-auto pb-10">
            <div className="px-4 py-2 text-xs font-bold text-text-secondary uppercase">
                Project Files
            </div>
            {rootEntries.map(entry => (
                <FileTreeItem
                    key={entry.name}
                    entry={entry}
                    path={rootPath}
                    onSelect={(p, isDir) => !isDir && onFileSelect(p)}
                />
            ))}
        </div>
    );
}
