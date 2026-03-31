import { useState, useRef, type FormEvent } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useProject, useUpdateProject, useDeleteProject } from '../hooks/useProjects';
import { useExportProject, useImportProject } from '../hooks/useImportExport';
import {
  Button, Input, TextArea, FormField, Section, EmptyState, Divider,
} from '../components/ui';
import styles from './ProjectSettings.module.css';

export default function ProjectSettings() {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const { data: project, isLoading } = useProject(slug);
  const updateProject = useUpdateProject(slug);
  const deleteProject = useDeleteProject();
  const exportProject = useExportProject(slug);
  const importProject = useImportProject(slug);

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [workingDir, setWorkingDir] = useState('');
  const [initialized, setInitialized] = useState(false);
  const [saved, setSaved] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importResult, setImportResult] = useState<Record<string, number> | null>(null);

  if (project && !initialized) {
    setName(project.name);
    setDescription(project.description ?? '');
    setWorkingDir(project.working_dir ?? '');
    setInitialized(true);
  }

  const handleSave = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSaved(false);
    updateProject.mutate(
      { name: name.trim(), description: description.trim() || undefined, working_dir: workingDir.trim() || undefined },
      {
        onSuccess: (updatedProject) => {
          setSaved(true);
          setTimeout(() => setSaved(false), 3000);
          if (updatedProject && updatedProject.slug !== slug) {
            navigate(`/projects/${updatedProject.slug}/settings`, { replace: true });
          }
        },
      }
    );
  };

  const handleDelete = () => {
    if (!slug) return;
    deleteProject.mutate(slug, { onSuccess: () => navigate('/', { replace: true }) });
  };

  if (isLoading) return <EmptyState message="Loading settings..." />;

  return (
    <div className={styles.page}>
      {/* General */}
      <form onSubmit={handleSave} className={styles.sectionGap}>
        <Section title="General">
          <div className={styles.fieldGroup}>
            <FormField label="Project Name">
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="Project name" />
            </FormField>
          </div>
          <div className={styles.fieldGroup}>
            <FormField label="Slug">
              <div className={styles.slugMono}>{slug}</div>
            </FormField>
          </div>
          <div className={styles.fieldGroup}>
            <FormField label="Description">
              <TextArea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Optional description..." rows={3} />
            </FormField>
          </div>
          <div className={styles.fieldGroup}>
            <FormField label="Working Directory" hint="Base directory for agent processes">
              <Input value={workingDir} onChange={(e) => setWorkingDir(e.target.value)} placeholder="/path/to/project" />
            </FormField>
          </div>
          {updateProject.error && <div className={styles.error}>{(updateProject.error as Error).message}</div>}
          {saved && <div className={styles.success}>Settings saved.</div>}
          <div className={styles.actions}>
            <Button type="submit" loading={updateProject.isPending}>Save Changes</Button>
          </div>
        </Section>
      </form>

      {/* Data Management */}
      <div className={styles.sectionGap}>
        <Section title="Data Management">
          <div className={styles.dataBlock}>
            <div className={styles.dataTitle}>Export Project Data</div>
            <div className={styles.dataDesc}>Download all data (rooms, issues, agents, knowledge base, templates, custom fields) as JSON.</div>
            <Button size="sm" onClick={() => exportProject.mutate()} loading={exportProject.isPending}>
              Export
            </Button>
            {exportProject.error && <div className={styles.error}>{(exportProject.error as Error).message}</div>}
          </div>

          <Divider />

          <div className={styles.dataBlock} style={{ marginTop: 'var(--space-lg)' }}>
            <div className={styles.dataTitle}>Import Project Data</div>
            <div className={styles.dataDesc}>Import from a previously exported JSON file.</div>
            <div className={styles.importRow}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                style={{ fontSize: 12 }}
                onChange={(e) => { setImportFile(e.target.files?.[0] ?? null); setImportResult(null); }}
              />
              <Button
                size="sm"
                disabled={!importFile}
                loading={importProject.isPending}
                onClick={() => {
                  if (!importFile) return;
                  setImportResult(null);
                  importProject.mutate(importFile, {
                    onSuccess: (result) => {
                      setImportResult(result);
                      setImportFile(null);
                      if (fileInputRef.current) fileInputRef.current.value = '';
                    },
                  });
                }}
              >
                Import
              </Button>
            </div>
            {importProject.error && <div className={styles.error}>{(importProject.error as Error).message}</div>}
            {importResult && (
              <div className={styles.success} style={{ marginTop: 'var(--space-sm)' }}>
                Import complete:{' '}
                {Object.entries(importResult)
                  .filter(([, count]) => count > 0)
                  .map(([key, count]) => `${count} ${key}`)
                  .join(', ') || 'no new data'}
              </div>
            )}
          </div>
        </Section>
      </div>

      {/* Danger Zone */}
      <div className={styles.sectionGap}>
        <Section title="Danger Zone" variant="danger">
          <div className={styles.dangerRow}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Delete this project</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                Permanently deletes all issues, chats, teams, and data.
              </div>
            </div>
            {!confirmDelete ? (
              <Button variant="danger" size="sm" onClick={() => setConfirmDelete(true)}>Delete Project</Button>
            ) : (
              <div className={styles.dangerActions}>
                <Button variant="secondary" size="sm" onClick={() => setConfirmDelete(false)}>Cancel</Button>
                <Button variant="danger" size="sm" onClick={handleDelete} loading={deleteProject.isPending}>
                  Confirm Delete
                </Button>
              </div>
            )}
          </div>
          {deleteProject.error && <div className={styles.error}>{(deleteProject.error as Error).message}</div>}
        </Section>
      </div>
    </div>
  );
}
