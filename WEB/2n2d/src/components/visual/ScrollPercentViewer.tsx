import React from 'react';
import {motion, useScroll} from 'motion/react';
import Style from './ScrollPercent.module.css';

const ScrollPercentViewer = ({ref, visible, loaded}: { ref: any, visible: boolean, loaded: boolean }) => {
    const {scrollYProgress} = useScroll({
        container: ref
    });

    if (!visible || !loaded)
        return "";

    return (
        <div className={Style.cont}>
            <div className={Style.cover}/>
            <motion.div
                style={{
                    scaleY: scrollYProgress,
                    position: 'absolute',
                    top: 0,
                    right: 0,
                    originY: 0
                }}
                className={Style.bar}
            />
        </div>
    );
};

export default ScrollPercentViewer;
