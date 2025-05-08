"use client";

import React, { useState } from "react";
import { usePathname } from "next/navigation";
import Styles from "./SideBar.module.css";
import { uploadONNX } from "@/lib/fileHandler/fileUpload";

const SideBar = () => {
  const [open, setOpen] = React.useState(false);
  const [result, setResult] = useState("");
  const pathname = usePathname();

  async function _uploadOnnx(e: any) {
    const result = await uploadONNX(e);
    setResult(JSON.stringify(result));
  }

  return (
    <div>
      <div
        className={
          open ? Styles.container : `${Styles.container} ${Styles.closed}`
        }
      >
        <a href={"/"}>
          <img
            src={open ? "logo2n2d.svg" : "logo.svg"}
            alt="logo"
            className={Styles.logo}
          />
        </a>
        <a
          href="/visualize"
          className={
            pathname === "/visualize"
              ? `${Styles.tabBut} ${Styles.active}`
              : Styles.tabBut
          }
        >
          <span className={Styles.iconWrapper}>
            <i className="fa-solid fa-chart-network"></i>
          </span>
          <span className={`${Styles.tabText}`}>Visualize</span>
        </a>
        <a
          href="/data"
          className={
            pathname === "/data"
              ? `${Styles.tabBut} ${Styles.active}`
              : Styles.tabBut
          }
        >
          <span className={Styles.iconWrapper}>
            <i className="fa-solid fa-chart-simple"></i>
          </span>
          <span className={`${Styles.tabText}`}>Data</span>
        </a>
        <a
          href="/optimize"
          className={
            pathname === "/optimize"
              ? `${Styles.tabBut} ${Styles.active}`
              : Styles.tabBut
          }
        >
          <span className={Styles.iconWrapper}>
            <i className="fa-solid fa-rabbit-running"></i>
          </span>
          <span className={`${Styles.tabText}`}>Optimization</span>
        </a>
        <button className={Styles.closeBut} onClick={() => setOpen(!open)}>
          {open ? "<<<" : ">"}
        </button>
        <div className={Styles.fileUploadContainer}>
          <label className="file-input-label" htmlFor={"onnx-input"}>
            <i className="fa-solid fa-upload"></i>
            {open ? "Upload ONNX Model" : ""}
            <input
              type="file"
              id="onnx-input"
              accept=".onnx"
              onChange={(e) => _uploadOnnx(e)}
            />
          </label>
        </div>
      </div>
    </div>
  );
};

export default SideBar;
