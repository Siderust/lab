import { AlertTriangle, Info } from "lucide-react";

interface Props {
  level: "info" | "warn" | "error";
  message: string;
}

export default function AlertCard({ level, message }: Props) {
  const styles = {
    info: "border-blue-700/40 bg-blue-950/30 text-blue-300",
    warn: "border-yellow-700/40 bg-yellow-950/30 text-yellow-300",
    error: "border-red-700/40 bg-red-950/30 text-red-300",
  };
  const Icon = level === "info" ? Info : AlertTriangle;

  return (
    <div className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-sm ${styles[level]}`}>
      <Icon className="h-4 w-4 mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}
