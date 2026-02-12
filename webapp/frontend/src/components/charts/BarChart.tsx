import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js-dist-min";
import { LIBRARY_COLORS } from "../../api/types";

interface Bar {
  library: string;
  value: number;
  /** Optional p95 value for error whiskers */
  upper?: number;
}

interface Props {
  bars: Bar[];
  yLabel?: string;
  title?: string;
}

/**
 * Grouped bar chart for ns/op (or throughput).
 */
export default function BarChart({
  bars,
  yLabel = "ns/op",
  title = "Performance (ns/op)",
}: Props) {
  if (bars.length === 0) {
    return (
      <div className="flex h-[280px] items-center justify-center rounded-xl border border-gray-800 text-sm text-gray-500">
        No performance data available.
      </div>
    );
  }

  const trace: Data = {
    x: bars.map((b) => b.library),
    y: bars.map((b) => b.value),
    type: "bar" as const,
    marker: {
      color: bars.map((b) => LIBRARY_COLORS[b.library] ?? "#9ca3af"),
    },
    error_y: bars.some((b) => b.upper != null)
      ? {
          type: "data" as const,
          array: bars.map((b) =>
            b.upper != null ? b.upper - b.value : 0
          ),
          visible: true,
          color: "#9ca3af",
        }
      : undefined,
  };

  const layout: Partial<Layout> = {
    title: { text: title, font: { color: "#e5e7eb", size: 14 } },
    xaxis: { color: "#9ca3af" },
    yaxis: {
      title: { text: yLabel },
      color: "#9ca3af",
      gridcolor: "#374151",
      zerolinecolor: "#374151",
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(17,24,39,0.6)",
    font: { color: "#d1d5db" },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 300,
  };

  return (
    <Plot
      data={[trace]}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      className="w-full"
    />
  );
}
