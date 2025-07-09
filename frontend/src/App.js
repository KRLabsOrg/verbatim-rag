import React from 'react';
import { ApiProvider } from './contexts/ApiContext';
import { DocumentsProvider } from './contexts/DocumentsContext';
import ImprovedFactInterface from './components/ImprovedFactInterface';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <ApiProvider>
        <DocumentsProvider>
          <ImprovedFactInterface />
        </DocumentsProvider>
      </ApiProvider>
    </ErrorBoundary>
  );
}

export default App;