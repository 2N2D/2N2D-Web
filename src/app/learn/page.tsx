"use client"
import react from 'react';
import "./styles.css"
import ParticleNetwork from "@/components/visual/particleNetwork";
import {motion} from "framer-motion";
import CoursesDisplayer from "@/components/courses/coursesDisplayer";
import { Trans } from '@lingui/react/macro';

export default function Learn() {
    return <motion.div className={"learnContainer"} transition={{delay: 0.4, duration: 0.2, ease: "easeOut"}}
                       initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}}>
        <div className={"neuroVision"}>
            <div className={"flex flex-col gap-[1rem] "}>
                <div>
                    <h2 className={"subtitle"}>Know nothing about neural networks?</h2>
                    <h3>NeuroVision <Trans>offers free courses about neural networks and how they work.</Trans></h3>
                    <h3><Trans>The Neural Network Playground offers an interactive way to see and understand how a neural
                        network
                        thinks and functions. Just click the button below to get started.</Trans></h3>
                </div>
                <a href={"https://neuro-vision-one.vercel.app"}>NeuralVision <i
                    className="fa-solid fa-arrow-up-right-from-square"></i></a>
            </div>
            <div className={"neuroLogo"}>
                <h1>NeuroVision</h1>
                <img src={"/neuroVisionLogo.png"} alt={"Neurovision Logo"}/>
            </div>
        </div>
        <div className={"visionFade"}/>
        <div className={"NNDev"}>
            <ParticleNetwork/>
            <div>
                <h2 className={"subtitle"}><Trans>Got the basics?</Trans></h2>
                <h3><Trans>If you got the strings about how neural networks work, then use the following guide to get
                    started creating your own. Don't forget to ask the AI Assistant for questions!</Trans></h3>
            </div>
            <img src={"/logo2n2dNNDEV.png"} alt={"logo"}/>
        </div>
        <div className={"NNDevFade"}/>
        <CoursesDisplayer/>
    </motion.div>
}