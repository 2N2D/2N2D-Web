'use client';
import React, {useRef, useEffect} from 'react';

const ParticleNetwork: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mouse = useRef({x: null, y: null});

    useEffect(() => {
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext('2d')!;
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        let particles: Particle[] = [];

        class Particle {
            x: number;
            y: number;
            vx: number;
            vy: number;
            radius: number;

            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.vx = (Math.random() - 0.5) * 1;
                this.vy = (Math.random() - 0.5) * 1;
                this.radius = 2;
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = '#fff';
                ctx.fill();
            }
        }

        function connect() {
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 100) {
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            particles.forEach(p => {
                p.update();
                p.draw();
            });

            // Mouse particle
            if (mouse.current.x !== null && mouse.current.y !== null) {
                ctx.beginPath();
                ctx.arc(mouse.current.x, mouse.current.y, 3, 0, Math.PI * 2);
                ctx.fillStyle = '#fff';
                ctx.fill();
                particles.forEach(p => {
                    const dx = p.x - mouse.current.x!;
                    const dy = p.y - mouse.current.y!;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 150) {
                        ctx.strokeStyle = 'rgba(255,255,255,0.3 )';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(mouse.current.x!, mouse.current.y!);
                        ctx.stroke();
                    }
                });
            }

            connect();
            requestAnimationFrame(animate);
        }

        for (let i = 0; i < 100; i++) {
            particles.push(new Particle());
        }

        window.addEventListener('mousemove', e => {
            // @ts-ignore
            mouse.current.x = e.clientX;
            // @ts-ignore
            mouse.current.y = e.clientY;
        });

        window.addEventListener('mouseout', () => {
            mouse.current.x = null;
            mouse.current.y = null;
        });

        animate();
    }, []);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                zIndex: -1,
                backgroundColor: 'black',
            }}
        />
    );
};

export default ParticleNetwork;