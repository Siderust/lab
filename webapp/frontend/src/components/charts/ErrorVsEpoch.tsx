import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js-dist-min";
import { LIBRARY_COLORS } from "../../api/types";

interface Series {
  library: string;
  epochs: number[];  // JD
  errors: number[];
}

interface Props {
  series: Series[];
  yLabel?: string;
  title?: string;
}

/**
 * Scatter plot of error vs epoch (JD).
 */
export default function ErrorVsEpoch({
  series,
  yLabel = "Error",
  title = "Error vs Epoch",
}: Props) {
  const traces: Data[] = series.map((s) => ({
    x: s.epochs,
    y: s.errors,
    type: "scatter" as const,
    mode: "markers" as const,
    name: s.library,
    marker: {
      color: LIBRARY_COLORS[s.library] ?? "#9ca3af",
      size: 4,
      opacity: 0.7,
    },
  }));

  const layout: Partial<Layout> = {
    title: { text: title, font: { color: "#e5e7eb", size: 14 } },
    xaxis: {
      title: { text: "JD (TT)" },
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
    legend: { bgcolor: "rgba(0,0,0,0)", font: { color: "#d1d5db" } },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 350,
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
