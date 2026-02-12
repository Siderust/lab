import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js-dist-min";
import { LIBRARY_COLORS } from "../../api/types";

interface Point {
  library: string;
  error: number;   // p99 or max error
  latency: number;  // ns/op
}

interface Props {
  points: Point[];
  xLabel?: string;
  yLabel?: string;
  title?: string;
}

/**
 * Pareto plot: error (x) vs latency (y), log-log, one point per library.
 */
export default function ParetoPlot({
  points,
  xLabel = "p99 Error",
  yLabel = "ns/op",
  title = "Accuracy vs Performance",
}: Props) {
  if (points.length === 0) {
    return (
      <div className="flex h-[280px] items-center justify-center rounded-xl border border-gray-800 text-sm text-gray-500">
        No performance data available for Pareto plot.
      </div>
    );
  }

  const traces: Data[] = points.map((p) => ({
    x: [p.error],
    y: [p.latency],
    type: "scatter" as const,
    mode: "text+markers" as const,
    name: p.library,
    text: [p.library],
    textposition: "top center" as const,
    marker: {
      color: LIBRARY_COLORS[p.library] ?? "#9ca3af",
      size: 12,
    },
    textfont: { color: LIBRARY_COLORS[p.library] ?? "#9ca3af", size: 11 },
  } as Data));

  const layout: Partial<Layout> = {
    title: { text: title, font: { color: "#e5e7eb", size: 14 } },
    xaxis: {
      title: { text: xLabel },
      type: "log",
      color: "#9ca3af",
      gridcolor: "#374151",
      zerolinecolor: "#374151",
    },
    yaxis: {
      title: { text: yLabel },
      type: "log",
      color: "#9ca3af",
      gridcolor: "#374151",
      zerolinecolor: "#374151",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(17,24,39,0.6)",
    font: { color: "#d1d5db" },
    showlegend: false,
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 300,
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      className="w-full"
    />
  );
}
