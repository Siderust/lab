import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Sidebar from "./components/layout/Sidebar";
import RunsList from "./pages/RunsList";
import RunOverview from "./pages/RunOverview";
import ExperimentDetail from "./pages/ExperimentDetail";
import CompareRuns from "./pages/CompareRuns";
import RunBenchmarks from "./pages/RunBenchmarks";
import PerformanceMatrix from "./pages/PerformanceMatrix";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 ml-56 p-8">
            <Routes>
              <Route path="/" element={<RunsList />} />
              <Route path="/runs/:runId" element={<RunOverview />} />
              <Route
                path="/runs/:runId/experiments/:experiment"
                element={<ExperimentDetail />}
              />
              <Route path="/compare" element={<CompareRuns />} />
              <Route path="/benchmark" element={<RunBenchmarks />} />
              <Route
                path="/runs/:runId/performance-matrix"
                element={<PerformanceMatrix />}
              />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
