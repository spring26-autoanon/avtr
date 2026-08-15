import { useState, useEffect, useRef } from "react";
import { colors } from "../../../../theme/colors";
import { AudioDiagSample } from "../../hooks/useServerAudio";

type ServerAudioStatsProps = {
  getAudioStats: React.MutableRefObject<
    () => {
      playedAudioDuration: number;
      missedAudioDuration: number;
      totalAudioMessages: number;
      delay: number;
      minPlaybackDelay: number;
      maxPlaybackDelay: number;
    }
  >;
  getAudioDiagLog?: React.MutableRefObject<() => AudioDiagSample[]>;
};

/**
 * Downloads the accumulated jitter-buffer diagnostic (see useServerAudio.ts's
 * AudioDiagSample) as a JSON file, for manual placement alongside the
 * server-side session logs (demo/sessions/<id>/) that feed the same
 * analysis workflow.
 */
const downloadAudioDiagLog = (samples: AudioDiagSample[]) => {
  const blob = new Blob([JSON.stringify(samples, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `client_audio_diag_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const ServerAudioStats = ({ getAudioStats, getAudioDiagLog }: ServerAudioStatsProps) => {
  const [audioStats, setAudioStats] = useState(getAudioStats.current());

  const movingAverageSum = useRef<number>(0.);
  const movingAverageCount = useRef<number>(0.);
  const movingBeta = 0.85;

  const convertMinSecs = (total_secs: number) => {
    if (!Number.isFinite(total_secs) || total_secs < 0) {
      total_secs = 0;
    }
    // convert secs to the format mm:ss.cc
    let mins = (Math.floor(total_secs / 60)).toString();
    let secs = (Math.floor(total_secs) % 60).toString();
    let cents = (Math.floor(100 * (total_secs - Math.floor(total_secs)))).toString();
    if (secs.length < 2) {
      secs = "0" + secs;
    }
    if (cents.length < 2) {
      cents = "0" + cents;
    }
    return mins + ":" + secs + "." + cents;
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const newAudioStats = getAudioStats.current();
      setAudioStats(newAudioStats);
      movingAverageCount.current *= movingBeta;
      movingAverageCount.current += (1 - movingBeta) * 1;
      movingAverageSum.current *= movingBeta;
      movingAverageSum.current += (1 - movingBeta) * newAudioStats.delay;

    }, 141);
    return () => {
      clearInterval(interval);
    };
  }, []);

  const movingLatency =
    movingAverageCount.current > 0
      ? movingAverageSum.current / movingAverageCount.current
      : audioStats.delay;

  return (
    <div className="w-full text-sm" style={{ color: colors.textPrimary }}>
      <table>
        <tbody>
          <tr>
            <td className="text-sm pr-2">Audio played: </td>
            <td>{convertMinSecs(audioStats.playedAudioDuration)}</td>
          </tr>
          <tr>
            <td className="text-sm pr-2">Missed audio: </td>
            <td>{convertMinSecs(audioStats.missedAudioDuration)}</td>
          </tr>
          <tr>
            <td className="text-sm pr-2">Latency: </td>
            <td>{Number.isFinite(movingLatency) ? movingLatency.toFixed(3) : "0.000"}</td>
          </tr>
          <tr>
            <td className="text-sm pr-2">Min/Max buffer: </td>
            <td>{audioStats.minPlaybackDelay.toFixed(3)} / {audioStats.maxPlaybackDelay.toFixed(3)}</td>
          </tr>
        </tbody>
      </table>
      {getAudioDiagLog && (
        <button
          onClick={() => downloadAudioDiagLog(getAudioDiagLog.current())}
          className="mt-2 text-xs underline hover:opacity-80"
          style={{ color: colors.textPrimary }}
        >
          Download audio diagnostics
        </button>
      )}
    </div>
  );
};
