import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import AgentRegistry from './pages/AgentRegistry';
import ProjectView from './pages/ProjectView';
import ProjectOverview from './pages/ProjectOverview';
import ChatRoom from './pages/ChatRoom';
import TaskBoard from './pages/TaskBoard';
import TaskDetail from './pages/TaskDetail';
import AgentManager from './pages/AgentManager';
import ProjectSettings from './pages/ProjectSettings';
import CustomFieldsAdmin from './pages/CustomFieldsAdmin';
import TemplatesPage from './pages/TemplatesPage';
import DocumentsPage from './pages/DocumentsPage';
import KnowledgeBase from './pages/KnowledgeBase';
import KBEditor from './pages/KBEditor';
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/agents" element={<AgentRegistry />} />
          <Route path="/custom-fields" element={<CustomFieldsAdmin />} />
          <Route path="/templates" element={<TemplatesPage />} />
          <Route path="/projects/:slug" element={<ProjectView />}>
            <Route index element={<Navigate to="overview" replace />} />
            <Route path="overview" element={<ProjectOverview />} />
            <Route path="chat" element={<ChatRoom />} />
            <Route path="chat/:room" element={<ChatRoom />} />
            <Route path="issues" element={<TaskBoard />} />
            <Route path="issues/:number" element={<TaskDetail />} />
            <Route path="agents" element={<AgentManager />} />
            <Route path="documents" element={<DocumentsPage />} />
            <Route path="kb" element={<KnowledgeBase />} />
            <Route path="kb/new" element={<KBEditor />} />
            <Route path="kb/edit/*" element={<KBEditor />} />
            <Route path="terminals" element={null} />
            <Route path="settings" element={<ProjectSettings />} />
          </Route>
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
