'use client';
import React, { useState, useEffect, FormEvent } from 'react';
import { startOptimization } from '@/lib/2n2dAPI';
import { downloadFileRequest } from '@/lib/feHandler';
import { getSessionTokenHash } from '@/lib/auth/authentication';
import { getSession } from '@/lib/sessionHandling/sessionManager';
import './styles.css';
import ONNXUploader from '@/components/fileUploadElements/ONNXUploader';
import CSVUploader from '@/components/fileUploadElements/CSVUploader';
import { deleteCsv, deleteOnnx } from '@/lib/sessionHandling/sessionUpdater';
import { motion } from 'framer-motion';

function Optimize() {
  const [features, setFeatures] = useState<string[]>([]);
  const [status, setStatus] = useState<string>('');
  const [progress, setProgress] = useState<number>(-1);
  const [csvFileName, setCsvFileName] = useState<string>('');
  const [onnxFileName, setOnnxFileName] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [downloading, setDownloading] = useState<boolean>(false);
  const [alert, setAlert] = useState<string | null>(null);

  async function statusUpdate() {
    const eventSource = new EventSource(
      `${process.env.NEXT_PUBLIC_TWONTWOD_ENDPOINT}/optimization-status/${await getSessionTokenHash()}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data.status);
      setProgress(data.progress);
      console.log('Progress update:', data);
    };

    eventSource.onerror = (err) => {
      console.log('SSE error:', err);
      // setAlert("Error connecting to server");
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }

  async function loadSession() {
    if (sessionStorage.getItem('currentSessionId')) {
      const session = await getSession(
        parseInt(sessionStorage.getItem('currentSessionId')!)
      );
      console.log(JSON.stringify(session.optResult));
      if (
        session &&
        session.optResult &&
        JSON.stringify(session.optResult).length > 2 &&
        JSON.stringify(session.optResult) != 'null' &&
        (session.optResult as any).best_config
      ) {
        setResult(session?.optResult);
        setProgress(100);
        setStatus('Optimization finished');
      } else {
        setResult(null);
        setProgress(-1);
      }

      setOnnxFileName(session?.onnxName!);
      setCsvFileName(session?.csvName!);
    }
  }

  function populateLists() {
    setFeatures([]);

    if (sessionStorage.getItem('modelName'))
      setOnnxFileName(sessionStorage.getItem('modelName')!);

    if (
      !sessionStorage.getItem('csvData') ||
      !sessionStorage.getItem('modelData')
    ) {
      return;
    }

    if (
      sessionStorage.getItem('modelData')!.length < 4 ||
      sessionStorage.getItem('csvData')!.length < 4
    )
      return;

    let csv = JSON.parse(sessionStorage.getItem('csvData')!);

    console.log(JSON.stringify(result));

    setCsvFileName(csv.summary.filename);

    let src = csv!.data;
    let feat = [];
    for (let key in src[0]) feat.push(key);

    setFeatures(feat);
  }

  useEffect(() => {
    populateLists();
    loadSession();
  }, []);

  async function optimize(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();

    setAlert(null);

    const formData = new FormData(e.currentTarget);
    const featuresaux = formData.getAll('sIFeatures[]');
    let Ifeatures: String[] = [];
    featuresaux.forEach((feat) => {
      Ifeatures.push(feat.toString());
    });
    const target = formData.get('target')?.toString()!;
    const encoding = formData.get('encoding')?.toString();
    const maxEpochs = Number(formData.get('epochs'));

    const sesId = sessionStorage.getItem('currentSessionId');
    if (!sesId) return;
    const session = await getSession(parseInt(sesId));
    if (!session) return;

    setProgress(0);
    setStatus('Starting optimization...');
    statusUpdate();

    const _result = await startOptimization(
      Ifeatures,
      target,
      maxEpochs,
      parseInt(sesId),
      session.csvUrl!,
      session.onnxUrl!,
      encoding!
    );

    if (typeof _result === 'string') {
      setAlert(_result);
      setProgress(-1);
      setStatus('Error: ' + _result);
    }

    setResult(_result);
    setProgress(100);
    setStatus('Optimization finished');
    console.log(_result);
    loadSession();
  }

  async function downloadOptimized() {
    setDownloading(true);
    if (!result) return;

    let fileName = onnxFileName.split('.')[0] + '_optimized.onnx';
    console.log(result.url);
    await downloadFileRequest(result.url!, 'rezult', fileName);
    setDownloading(false);
  }

  return (
    <motion.main
      className={'pageOpt'}
      transition={{
        delay: 0.4,
        duration: 0.2,
        ease: 'easeOut',
        staggerChildren: 0.1
      }}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className={'flex w-full gap-[0.1rem]'}>
        <div className={'flex flex-col'} style={{ width: '100%' }}>
          <form
            className={'optimizeForm area'}
            style={{ width: '100%' }}
            onSubmit={optimize}
          >
            <h2 className={'subtitle'}>Optimization settings</h2>
            <div className={'formGroup'}>
              <div className={'element'}>
                <label>Input Features:</label>
                <ul className={'featuresList'}>
                  {features.length == 0 ? (
                    <li>Upload a CSV dataset to see available features</li>
                  ) : (
                    features.map((feat, i) => (
                      <li key={i}>
                        <input
                          type='checkbox'
                          value={feat}
                          name={'sIFeatures[]'}
                        />
                        <p>{feat}</p>
                      </li>
                    ))
                  )}
                </ul>
              </div>
              <div className={'element'}>
                <label>Target Feature:</label>
                <select className={'targetFeature'} name={'target'}>
                  {features.length == 0 ? (
                    <option disabled>
                      Upload a CSV dataset to see available features
                    </option>
                  ) : (
                    features.map((feat, i) => (
                      <option key={i} value={feat}>
                        {feat}
                      </option>
                    ))
                  )}
                </select>
                <label>Encoding type:</label>
                <select className={'targetFeature'} name={'encoding'}>
                  <option>None</option>
                  <option>label</option>
                  <option>onehot</option>
                </select>
              </div>
              <div className={'element'}>
                <label>Maximum Epochs Per Configuration:</label>
                <input type='number' name={'epochs'} defaultValue='10' />
              </div>
            </div>

            <input
              type='submit'
              id='opt-start-optimization'
              value='Start Optimization'
              disabled={
                !!(
                  progress > -1 &&
                  progress < 100 &&
                  onnxFileName &&
                  csvFileName
                )
              }
            />
            {alert != null ? (
              <div className={'alert'}>
                <h1>{alert}</h1>
              </div>
            ) : (
              ''
            )}
          </form>
          <div
            style={progress != -1 ? { width: '100%' } : { width: 0 }}
            className={`progressZone ${progress != -1 ? 'area' : ''}`}
          >
            <h1>{status}</h1>
            {/*<div className={progress == 100 || progress == -1 ? "hidden" : "loaderWrapper"}>*/}
            {/*    <span className="loader"></span>*/}
            {/*</div>*/}
            <div className={'barWrapper'}>
              <p>{progress}%</p>
              <div className={'loadingBar'}>
                <div
                  className={`barFiller`}
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div className={'dataArea area vertical'}>
            <h1>
              <b>ONNX file:</b> {onnxFileName}
            </h1>
            <div className={'flex gap-[1rem]'}>
              <ONNXUploader callBack={populateLists} />
              <button
                className={'deleteButton'}
                onClick={async () => {
                  const curSesId = sessionStorage.getItem('currentSessionId');
                  if (!curSesId) return;
                  await deleteOnnx(parseInt(curSesId));

                  sessionStorage.removeItem('modelResponse');
                  sessionStorage.removeItem('modelData');
                  sessionStorage.removeItem('modelName');

                  await loadSession();
                  populateLists();
                }}
              >
                Clear Data <i className='fa-solid fa-trash-xmark'></i>
              </button>
            </div>
          </div>
          <div className={'dataArea area vertical'}>
            <h1>
              <b>CSV file:</b> {csvFileName}
            </h1>
            <div className={'flex gap-[1rem]'}>
              <CSVUploader callBack={populateLists} />
              <button
                className={'deleteButton'}
                onClick={async () => {
                  const curSesId = sessionStorage.getItem('currentSessionId');
                  if (!curSesId) return;
                  await deleteCsv(parseInt(curSesId));

                  sessionStorage.removeItem('csvData');

                  await loadSession();
                  populateLists();
                }}
              >
                Clear Data <i className='fa-solid fa-trash-xmark'></i>
              </button>
            </div>
          </div>
        </div>
        <div className={'titleArea'}>
          <h1 className={'dataTitle title'}>Optimization</h1>
        </div>
      </div>
      {result &&
      JSON.stringify(result).length > 2 &&
      JSON.stringify(result) != 'null' &&
      true &&
      result &&
      result.best_config ? (
        <motion.div
          className={`resultArea ${progress == 100 ? 'area' : ''}`}
          style={progress == 100 ? { height: '100%' } : { height: 0 }}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <h1 className={'main subtitle'}>Optimization results:</h1>
          <div className={'flex flex-col gap-[1rem] p-[1rem]'}>
            <h2 className={'subtitle'}>Best configuration:</h2>
            <div className={'result'}>
              <div className={'info'}>
                <h2>Neurons:</h2> {result?.best_config.neurons}
              </div>
              <div className={'info'}>
                <h2>Layers:</h2> {result?.best_config.layers}
              </div>
              <div className={'info'}>
                <h2>Test loss:</h2> {result?.best_config.test_loss}
              </div>
              <div className={'info'}>
                <h2>R2 score:</h2> {result?.best_config.r2_score}
              </div>
            </div>
          </div>
          <table className={'table'}>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Neurons</th>
                <th>Layers</th>
                <th>Test loss:</th>
                <th>R2 score:</th>
              </tr>
              <tr>
                <td>Best config</td>
                <td>{result?.best_config.neurons}</td>
                <td>{result?.best_config.layers}</td>
                <td>{result?.best_config.test_loss}</td>
                <td>{result?.best_config.r2_score}</td>
              </tr>
            </thead>
            <tbody>
              {result?.results.map(
                (
                  res: {
                    neurons:
                      | string
                      | number
                      | bigint
                      | boolean
                      | React.ReactElement<
                          unknown,
                          string | React.JSXElementConstructor<any>
                        >
                      | Iterable<React.ReactNode>
                      | React.ReactPortal
                      | Promise<
                          | string
                          | number
                          | bigint
                          | boolean
                          | React.ReactPortal
                          | React.ReactElement<
                              unknown,
                              string | React.JSXElementConstructor<any>
                            >
                          | Iterable<React.ReactNode>
                          | null
                          | undefined
                        >
                      | null
                      | undefined;
                    layers:
                      | string
                      | number
                      | bigint
                      | boolean
                      | React.ReactElement<
                          unknown,
                          string | React.JSXElementConstructor<any>
                        >
                      | Iterable<React.ReactNode>
                      | React.ReactPortal
                      | Promise<
                          | string
                          | number
                          | bigint
                          | boolean
                          | React.ReactPortal
                          | React.ReactElement<
                              unknown,
                              string | React.JSXElementConstructor<any>
                            >
                          | Iterable<React.ReactNode>
                          | null
                          | undefined
                        >
                      | null
                      | undefined;
                    test_loss:
                      | string
                      | number
                      | bigint
                      | boolean
                      | React.ReactElement<
                          unknown,
                          string | React.JSXElementConstructor<any>
                        >
                      | Iterable<React.ReactNode>
                      | React.ReactPortal
                      | Promise<
                          | string
                          | number
                          | bigint
                          | boolean
                          | React.ReactPortal
                          | React.ReactElement<
                              unknown,
                              string | React.JSXElementConstructor<any>
                            >
                          | Iterable<React.ReactNode>
                          | null
                          | undefined
                        >
                      | null
                      | undefined;
                    r2_score:
                      | string
                      | number
                      | bigint
                      | boolean
                      | React.ReactElement<
                          unknown,
                          string | React.JSXElementConstructor<any>
                        >
                      | Iterable<React.ReactNode>
                      | React.ReactPortal
                      | Promise<
                          | string
                          | number
                          | bigint
                          | boolean
                          | React.ReactPortal
                          | React.ReactElement<
                              unknown,
                              string | React.JSXElementConstructor<any>
                            >
                          | Iterable<React.ReactNode>
                          | null
                          | undefined
                        >
                      | null
                      | undefined;
                  },
                  i: number
                ) => (
                  <tr key={i}>
                    <td>Config {i + 1}</td>
                    <td>{res.neurons}</td>
                    <td>{res.layers}</td>
                    <td>{res.test_loss}</td>
                    <td>{res.r2_score}</td>
                  </tr>
                )
              )}
            </tbody>
          </table>

          <button onClick={downloadOptimized} disabled={downloading}>
            {downloading ? 'Downloading...' : 'Download optimized'}{' '}
            <i className='fa-solid fa-file-arrow-down'></i>
          </button>
        </motion.div>
      ) : (
        ''
      )}
    </motion.main>
  );
}

export default Optimize;
