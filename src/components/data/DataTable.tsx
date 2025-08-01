'use client';

import React, { useState } from 'react';
import Style from './DataTable.module.css';
import { Trans, useLingui } from '@lingui/react/macro';

function Table(results: any) {
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState<number>(10);

  const { t } = useLingui();

  results = results.result;
  if (results === null || results == undefined) {
    return (
      <div>
        <h1>
          <Trans>No data to display</Trans>
        </h1>
      </div>
    );
  }

  let src = results.data;
  let data: Map<string, Array<any>> = new Map();

  for (let i = 0; i < src.length; i++) {
    for (let key in src[i]) {
      const value = src[i][key];
      data.has(key) ? data.get(key)!.push(value) : data.set(key, [value]);
    }
  }

  const maxRows = Math.max(...[...data.values()].map((arr) => arr.length));

  const totalPages = Math.ceil(maxRows / rowsPerPage);

  const currentRows = [...Array(rowsPerPage)].map((_, rowIndex) =>
    rowIndex + (currentPage - 1) * rowsPerPage < maxRows
      ? rowIndex + (currentPage - 1) * rowsPerPage
      : -1
  );

  return (
    <div>
      <div className={Style.controlsContainer}>
        <div className={Style.controls}>
          <button
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
          >
            <i className='fa-solid fa-left-to-line'></i>
          </button>
          <button
            onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
          >
            <i className='fa-solid fa-left'></i>
          </button>
          <span>
            <Trans>
              Page {currentPage} of {totalPages}
            </Trans>
          </span>
          <button
            onClick={() =>
              setCurrentPage((next) => Math.min(next + 1, totalPages))
            }
            disabled={currentPage === totalPages}
          >
            <i className='fa-solid fa-right'></i>
          </button>
          <button
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage === totalPages}
          >
            <i className='fa-solid fa-right-to-line'></i>
          </button>
        </div>
        <div className={Style.controls}>
          <Trans>Rows/Page:</Trans>
          <input
            type={'number'}
            min={'1'}
            value={rowsPerPage}
            onChange={(e) => {
              setRowsPerPage(Number(e.target.value));
              if (rowsPerPage < 1) {
                setRowsPerPage(1);
              }
            }}
          />
        </div>
      </div>
      <table className={Style.table}>
        <thead>
          <tr className={Style.headerRow}>
            {[...data.keys()].map((key, index) => (
              <th key={index}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {currentRows.map((rowIndex, rowIndexInPage) => {
            if (rowIndex === -1) return null;

            return (
              <tr key={rowIndexInPage}>
                {[...data].map(([key, values], colIndex) => (
                  <td key={colIndex}>{values[rowIndex] || '-'}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      {rowsPerPage < 30 ? (
        ''
      ) : (
        <div className={Style.controlsContainer}>
          <div className={Style.controls}>
            <button
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
            >
              <i className='fa-solid fa-left-to-line'></i>
            </button>
            <button
              onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
            >
              <i className='fa-solid fa-left'></i>
            </button>
            <span>
              <Trans>
                Page {currentPage} of {totalPages}
              </Trans>
            </span>
            <button
              onClick={() =>
                setCurrentPage((next) => Math.min(next + 1, totalPages))
              }
              disabled={currentPage === totalPages}
            >
              <i className='fa-solid fa-right'></i>
            </button>
            <button
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
            >
              <i className='fa-solid fa-right-to-line'></i>
            </button>
          </div>
          <div className={Style.controls}>
            <Trans>Rows/Page:</Trans>
            <input
              type={'number'}
              value={rowsPerPage}
              min={'1'}
              onChange={(e) => {
                setRowsPerPage(Number(e.target.value));
                if (rowsPerPage < 1) {
                  setRowsPerPage(1);
                }
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default Table;
