import JSZip from 'jszip';

/**
 * Download a single file via the browser by creating a temporary anchor element.
 */
export function downloadFile(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Bundle multiple files into a .zip archive and download it via the browser.
 */
export async function downloadZip(
  zipName: string,
  files: { name: string; content: string }[]
): Promise<void> {
  const zip = new JSZip();

  for (const file of files) {
    zip.file(file.name, file.content);
  }

  const blob = await zip.generateAsync({ type: 'blob' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = zipName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Check if the File System Access API is available (showDirectoryPicker).
 */
export function hasFileSystemAccess(): boolean {
  return typeof window !== 'undefined' && 'showDirectoryPicker' in window;
}

/**
 * Save files directly to disk using the File System Access API.
 *
 * Opens a directory picker, then writes each file into the chosen directory.
 * Returns the number of files written, or null if the user cancelled.
 */
export async function saveFilesToDisk(
  files: { name: string; content: string }[]
): Promise<number | null> {
  if (!hasFileSystemAccess()) {
    throw new Error('File System Access API is not supported in this browser');
  }

  let dirHandle: FileSystemDirectoryHandle;
  try {
    dirHandle = await window.showDirectoryPicker({
      mode: 'readwrite',
      startIn: 'desktop',
    });
  } catch {
    // User cancelled the picker
    return null;
  }

  let written = 0;
  for (const file of files) {
    const fileHandle = await dirHandle.getFileHandle(file.name, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(file.content);
    await writable.close();
    written++;
  }

  return written;
}
