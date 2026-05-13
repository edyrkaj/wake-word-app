import React, { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web/wasm';
import { computeLogMel, N_MELS, N_FRAMES } from './mel';
import './App.css';

const WAKE_WORD_THRESHOLD = 0.9;
// UX copy; detection follows your trained ONNX. Dataset folder e.g. `hey_yeli` (see record_dataset / train_model --positive-label).
const WAKE_WORD_PHRASE = 'hey Yeli';

const modelUrl = () =>
  process.env.NODE_ENV === 'development'
    ? `/model_tiny.onnx?nocache=${Date.now()}`
    : '/model_tiny.onnx';

const processorWorkletUrl = () =>
  process.env.NODE_ENV === 'development'
    ? `/processor.js?v=${Date.now()}`
    : '/processor.js';

const WakeWordDetector = () => {
  const [isListening, setIsListening] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [wakeWordDetected, setWakeWordDetected] = useState(false);
  const [lastScore, setLastScore] = useState(null);
  const [logs, setLogs] = useState([]);
  const [transcript, setTranscript] = useState('');
  const sessionRef = useRef(null);
  const modelInitStartedRef = useRef(false);
  const audioCtxRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const recognitionRef = useRef(null);
  const recognitionActiveRef = useRef(false);
  const recognitionRestartTimerRef = useRef(null);
  const finalTranscriptRef = useRef('');
  const wakeWordDetectedRef = useRef(false);
  const chunkCountRef = useRef(0);
  const inferenceInFlightRef = useRef(false);
  const lastInferenceAtRef = useRef(0);

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((currentLogs) => [`${timestamp} ${message}`, ...currentLogs].slice(0, 30));
  };

  // Load Model
  useEffect(() => {
    // Guard against React 18 StrictMode double-invocation in dev.
    if (modelInitStartedRef.current) return;
    modelInitStartedRef.current = true;

    const initModel = async () => {
      try {
        addLog('Loading ONNX model...');
        sessionRef.current = await ort.InferenceSession.create(modelUrl(), {
          executionProviders: ['wasm'], // Use WASM for best compatibility
        });
        setModelReady(true);
        addLog('Model loaded.');
      } catch (error) {
        addLog(`Model load failed: ${error.message}`);
        console.error('Model load failed', error);
      }
    };
    initModel();
  }, []);

  const startListening = async () => {
    try {
      addLog('Requesting microphone access...');
      wakeWordDetectedRef.current = false;
      setWakeWordDetected(false);
      finalTranscriptRef.current = '';
      setTranscript('');
      audioCtxRef.current = new AudioContext({ sampleRate: 16000 });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const source = audioCtxRef.current.createMediaStreamSource(stream);
      sourceNodeRef.current = source;

      await audioCtxRef.current.audioWorklet.addModule(processorWorkletUrl());
      const processorNode = new AudioWorkletNode(audioCtxRef.current, 'audio-processor');
      processorNodeRef.current = processorNode;

      processorNode.port.onmessage = (event) => {
        chunkCountRef.current += 1;
        runInference(event.data);
      };

      source.connect(processorNode);
      setIsListening(true);
      addLog(`Microphone started. Waiting for "${WAKE_WORD_PHRASE}" score > ${WAKE_WORD_THRESHOLD}.`);
    } catch (error) {
      addLog(`Microphone failed: ${error.message}`);
      console.error('Microphone failed', error);
    }
  };

  const stopListening = () => {
    addLog('Stopping...');

    recognitionActiveRef.current = false;
    if (recognitionRestartTimerRef.current) {
      clearTimeout(recognitionRestartTimerRef.current);
      recognitionRestartTimerRef.current = null;
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (err) {
        addLog(`Recognition stop failed: ${err.message}`);
      }
      recognitionRef.current = null;
    }

    if (processorNodeRef.current) {
      try {
        processorNodeRef.current.port.onmessage = null;
        processorNodeRef.current.disconnect();
      } catch (err) {
        // disconnect on already-disconnected node is fine
      }
      processorNodeRef.current = null;
    }

    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.disconnect();
      } catch (err) {
        // ignore
      }
      sourceNodeRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    setIsListening(false);
    wakeWordDetectedRef.current = false;
    setWakeWordDetected(false);
    addLog('Stopped listening.');
  };

  const runInference = async (pcmData) => {
    if (!sessionRef.current || inferenceInFlightRef.current) return;
    inferenceInFlightRef.current = true;
    lastInferenceAtRef.current = Date.now();

    try {
      let sumSquares = 0;
      for (let i = 0; i < pcmData.length; i += 1) {
        sumSquares += pcmData[i] * pcmData[i];
      }
      const rms = Math.sqrt(sumSquares / pcmData.length);

      const features = computeLogMel(pcmData);
      const input = new ort.Tensor('float32', features, [1, 1, N_MELS, N_FRAMES]);
      const outputs = await sessionRef.current.run({ input });
      const score = outputs.output.data[1];
      setLastScore(score);
      addLog(`Audio #${chunkCountRef.current}, rms=${rms.toFixed(4)}, score=${score.toFixed(4)}`);

      if (!wakeWordDetectedRef.current && score > WAKE_WORD_THRESHOLD) {
        wakeWordDetectedRef.current = true;
        setWakeWordDetected(true);
        addLog(`Wake Word Detected! score=${score.toFixed(4)}. Starting transcript...`);
        console.log('Wake Word Detected!', score);
        startSpeechRecognition();
      }
    } catch (e) {
      addLog(`Inference failed: ${e.message}`);
      console.error('Inference failed', e);
    } finally {
      inferenceInFlightRef.current = false;
    }
  };

  const startSpeechRecognition = () => {
    if (recognitionActiveRef.current) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      addLog('Speech transcript unavailable in this browser.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    finalTranscriptRef.current = '';
    setTranscript('');

    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const chunk = result[0].transcript;
        if (result.isFinal) {
          finalTranscriptRef.current = (finalTranscriptRef.current + ' ' + chunk).trim();
          addLog(`Heard (final): "${chunk.trim()}"`);
        } else {
          interim += chunk;
        }
      }
      const combined = (finalTranscriptRef.current + ' ' + interim).trim();
      setTranscript(combined);
    };

    recognition.onerror = (event) => {
      addLog(`Speech recognition error: ${event.error}`);
      // These are fatal: don't keep restarting in a tight loop.
      if (
        event.error === 'not-allowed' ||
        event.error === 'service-not-allowed' ||
        event.error === 'aborted'
      ) {
        recognitionActiveRef.current = false;
      }
    };

    recognition.onend = () => {
      if (!recognitionActiveRef.current) return;
      // Defer restart so a fast-failing session can't pin the main thread.
      recognitionRestartTimerRef.current = setTimeout(() => {
        recognitionRestartTimerRef.current = null;
        if (!recognitionActiveRef.current) return;
        try {
          recognition.start();
          addLog('Speech transcript auto-restarted.');
        } catch (err) {
          addLog(`Speech transcript restart failed: ${err.message}`);
        }
      }, 300);
    };

    recognitionActiveRef.current = true;
    recognition.start();
    recognitionRef.current = recognition;
    addLog('Speech transcript started after wake word.');
  };

  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h1>Wake Word Demo</h1>
      <button onClick={startListening} disabled={isListening}>
        {isListening ? "Listening..." : "Start Microphone"}
      </button>
      <button
        onClick={stopListening}
        disabled={!isListening}
        style={{ marginLeft: '8px' }}
      >
        Stop
      </button>
      <div style={{ marginTop: '20px' }}>
        <p>Model: {modelReady ? 'ready' : 'loading'}</p>
        <p>Wake word: {wakeWordDetected ? 'detected, transcribing' : 'waiting'}</p>
        <p>Latest score: {lastScore === null ? '-' : lastScore.toFixed(4)}</p>
        <p>Transcript: {wakeWordDetected ? (transcript || '-') : `locked until ${WAKE_WORD_PHRASE}`}</p>
      </div>
      <pre style={{
        background: '#111',
        color: '#0f0',
        margin: '20px auto 0',
        maxWidth: '720px',
        minHeight: '240px',
        overflow: 'auto',
        padding: '12px',
        textAlign: 'left',
        whiteSpace: 'pre-wrap'
      }}>
        {logs.join('\n')}
      </pre>
      <div className="footer">create from <a href="https://e-soft.al" target="_blank" rel="noopener noreferrer">e-soft.al</a></div>
    </div>
  );
};

export default WakeWordDetector;