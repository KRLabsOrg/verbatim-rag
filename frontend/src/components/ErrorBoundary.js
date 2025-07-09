import React from 'react';
import { AlertTriangle, RefreshCw, RotateCcw } from 'lucide-react';
import { Button } from './ui/button.jsx';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card.jsx';
import { Badge } from './ui/badge.jsx';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-red-50 flex items-center justify-center p-4">
          <Card className="max-w-md w-full">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-100 rounded-lg">
                  <AlertTriangle className="w-6 h-6 text-red-600" />
                </div>
                <div>
                  <CardTitle className="text-lg">Something went wrong</CardTitle>
                  <Badge variant="destructive" className="mt-1">Error</Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                An unexpected error occurred. Please try refreshing the page or contact support if the problem persists.
              </p>
              
              <div className="flex gap-2">
                <Button 
                  onClick={() => window.location.reload()}
                  className="flex-1"
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh Page
                </Button>
                <Button 
                  variant="outline" 
                  onClick={() => this.setState({ hasError: false, error: null })}
                  className="flex-1"
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Try Again
                </Button>
              </div>
              
              {this.state.error && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm text-muted-foreground hover:text-foreground transition-colors">
                    Error Details
                  </summary>
                  <pre className="mt-2 text-xs bg-muted p-3 rounded overflow-auto max-h-32 border">
                    {this.state.error.toString()}
                  </pre>
                </details>
              )}
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;