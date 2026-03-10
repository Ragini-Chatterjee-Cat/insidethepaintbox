// Hyperplane Game - Transform data to make it linearly separable (3D version)
// Multi-level version with increasing difficulty

class HyperplaneGame {
    constructor() {
        // Level definitions with increasing difficulty
        this.levels = [
            {
                name: "Level 1: XOR Pattern",
                description: "Separate points where X and Y have the same sign from those with opposite signs.",
                hint1: "Look at where the dark and light points sit. What quadrants are they in? Is there a pattern to which quadrants have which color?",
                hint2: "Think about what makes two numbers 'agree' or 'disagree' in sign. How might you combine X and Y to capture that relationship?",
                hint3: "Consider: what simple operation on two numbers tells you whether they share the same sign?",
                // Answer: return [x * y];
                data: [
                    // Class 0 (Dark) - same sign quadrants
                    { x: 1, y: 1, class: 0 },
                    { x: 1.5, y: 1.2, class: 0 },
                    { x: 0.8, y: 1.3, class: 0 },
                    { x: -1, y: -1, class: 0 },
                    { x: -1.2, y: -0.8, class: 0 },
                    { x: -0.9, y: -1.1, class: 0 },
                    // Class 1 (Light) - opposite sign quadrants
                    { x: 1, y: -1, class: 1 },
                    { x: 1.2, y: -0.9, class: 1 },
                    { x: 0.9, y: -1.2, class: 1 },
                    { x: -1, y: 1, class: 1 },
                    { x: -0.8, y: 1.1, class: 1 },
                    { x: -1.1, y: 0.9, class: 1 },
                ]
            },
            {
                name: "Level 2: Circular Pattern",
                description: "Separate points near the origin from points far from the origin.",
                hint1: "Forget the axes for a moment. What matters here isn't left/right or up/down - it's something else entirely about each point's position.",
                hint2: "If you were standing at the origin, how would you describe the difference between dark and light points? What single measurement captures that?",
                hint3: "Distance from the center is key. How do you measure distance in 2D? You don't need the exact distance - a related quantity works too.",
                // Answer: return [x*x + y*y - 1.5];
                data: [
                    // Class 0 (Dark) - inner circle (close to origin)
                    { x: 0.3, y: 0.4, class: 0 },
                    { x: -0.5, y: 0.3, class: 0 },
                    { x: 0.4, y: -0.5, class: 0 },
                    { x: -0.3, y: -0.4, class: 0 },
                    { x: 0.6, y: 0.1, class: 0 },
                    { x: -0.2, y: 0.6, class: 0 },
                    // Class 1 (Light) - outer ring (far from origin)
                    { x: 1.5, y: 1.2, class: 1 },
                    { x: -1.4, y: 1.0, class: 1 },
                    { x: 1.3, y: -1.1, class: 1 },
                    { x: -1.2, y: -1.3, class: 1 },
                    { x: 1.8, y: 0.2, class: 1 },
                    { x: -0.3, y: -1.7, class: 1 },
                ]
            },
            {
                name: "Level 3: Hyperbolic Pattern",
                description: "A challenging pattern where points follow hyperbolic curves.",
                hint1: "Compare the dark and light points. Are the dark points more 'horizontal' or more 'vertical' in their position? What about the light ones?",
                hint2: "Ask yourself: is the point further from the X-axis or the Y-axis? Which coordinate dominates - and does that relate to the color?",
                hint3: "You've used addition of squares before. What happens if you use subtraction instead? How does that capture 'which axis dominates'?",
                // Answer: return [x*x - y*y];
                data: [
                    // Class 0 (Dark) - where |x| > |y| (horizontal hyperbola regions)
                    { x: 1.5, y: 0.3, class: 0 },
                    { x: 1.3, y: -0.4, class: 0 },
                    { x: -1.4, y: 0.2, class: 0 },
                    { x: -1.6, y: -0.3, class: 0 },
                    { x: 1.8, y: 0.5, class: 0 },
                    { x: -1.2, y: 0.1, class: 0 },
                    // Class 1 (Light) - where |y| > |x| (vertical hyperbola regions)
                    { x: 0.3, y: 1.4, class: 1 },
                    { x: -0.2, y: 1.6, class: 1 },
                    { x: 0.4, y: -1.3, class: 1 },
                    { x: -0.3, y: -1.5, class: 1 },
                    { x: 0.1, y: 1.2, class: 1 },
                    { x: -0.5, y: -1.7, class: 1 },
                ]
            }
        ];

        this.currentLevel = 0;
        this.levelsCompleted = [false, false, false];

        // Load progress from localStorage
        const savedProgress = localStorage.getItem('hyperplaneProgress');
        if (savedProgress) {
            const progress = JSON.parse(savedProgress);
            this.levelsCompleted = progress.levelsCompleted || [false, false, false];
            this.currentLevel = progress.currentLevel || 0;
        }

        this.originalData = this.levels[this.currentLevel].data;
        this.transformedData = [...this.originalData];
        this.isSeparable = false;
        this.hintLevel = 0;

        // 3D rotation angles
        this.rotationX = -20;
        this.rotationZ = 45;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Colors - dark brown and light cream (no red/green)
        this.darkColor = '#5c4033';  // Dark brown
        this.lightColor = '#e8d4b8'; // Light cream
        this.planeColor = 'rgba(139, 115, 85, 0.4)';
    }

    init() {
        this.setupCanvas();
        this.render();
        this.renderStats();
        this.renderDataTable();
        this.renderMessage();
        this.renderLevelIndicator();
    }

    loadLevel(levelIndex) {
        if (levelIndex < 0 || levelIndex >= this.levels.length) return;

        this.currentLevel = levelIndex;
        this.originalData = this.levels[levelIndex].data;
        this.transformedData = this.originalData.map(p => ({ ...p, tz: 0 }));
        this.isSeparable = false;
        this.hintLevel = 0;
        this.rotationX = -20;
        this.rotationZ = 45;

        const codeArea = document.getElementById('userCode');
        if (codeArea) {
            codeArea.value = '  // Your Z coordinate formula\n  // Example: return [x + y];\n  return [0];';
        }

        const errorMsg = document.getElementById('errorMessage');
        if (errorMsg) errorMsg.textContent = '';

        this.saveProgress();
        this.render();
        this.renderStats();
        this.renderDataTable();
        this.renderMessage();
        this.renderLevelIndicator();
    }

    saveProgress() {
        localStorage.setItem('hyperplaneProgress', JSON.stringify({
            levelsCompleted: this.levelsCompleted,
            currentLevel: this.currentLevel
        }));
    }

    renderLevelIndicator() {
        const container = document.getElementById('levelIndicator');
        if (!container) return;

        const level = this.levels[this.currentLevel];
        let html = `<div class="level-header">
            <h3>${level.name}</h3>
            <p class="level-description">${level.description}</p>
            <div class="level-progress">`;

        for (let i = 0; i < this.levels.length; i++) {
            const completed = this.levelsCompleted[i];
            const current = i === this.currentLevel;
            let statusClass = '';
            if (completed) statusClass = 'completed';
            else if (current) statusClass = 'current';

            html += `<span class="level-dot ${statusClass}" title="Level ${i + 1}${completed ? ' (Completed)' : ''}">${i + 1}</span>`;
        }

        html += `</div></div>`;
        container.innerHTML = html;
    }

    setupCanvas() {
        const canvas3d = document.getElementById('plot3d');
        if (canvas3d) {
            // Set canvas resolution based on display size
            this.resizeCanvas(canvas3d);

            // Handle window resize
            window.addEventListener('resize', () => {
                this.resizeCanvas(canvas3d);
                this.render();
            });

            // Mouse drag for rotation
            canvas3d.addEventListener('mousedown', (e) => {
                this.isDragging = true;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
            });

            canvas3d.addEventListener('mousemove', (e) => {
                if (this.isDragging) {
                    const deltaX = e.clientX - this.lastMouseX;
                    const deltaY = e.clientY - this.lastMouseY;
                    this.rotationZ += deltaX * 0.5;
                    this.rotationX += deltaY * 0.5;
                    this.rotationX = Math.max(-90, Math.min(90, this.rotationX));
                    this.lastMouseX = e.clientX;
                    this.lastMouseY = e.clientY;
                    this.render();
                }
            });

            canvas3d.addEventListener('mouseup', () => {
                this.isDragging = false;
            });

            canvas3d.addEventListener('mouseleave', () => {
                this.isDragging = false;
            });

            // Touch support
            canvas3d.addEventListener('touchstart', (e) => {
                this.isDragging = true;
                this.lastMouseX = e.touches[0].clientX;
                this.lastMouseY = e.touches[0].clientY;
            }, { passive: true });

            canvas3d.addEventListener('touchmove', (e) => {
                if (this.isDragging) {
                    const deltaX = e.touches[0].clientX - this.lastMouseX;
                    const deltaY = e.touches[0].clientY - this.lastMouseY;
                    this.rotationZ += deltaX * 0.5;
                    this.rotationX += deltaY * 0.5;
                    this.rotationX = Math.max(-90, Math.min(90, this.rotationX));
                    this.lastMouseX = e.touches[0].clientX;
                    this.lastMouseY = e.touches[0].clientY;
                    this.render();
                }
            }, { passive: true });

            canvas3d.addEventListener('touchend', () => {
                this.isDragging = false;
            });
        }
    }

    resizeCanvas(canvas) {
        // Get the displayed size of the canvas
        const rect = canvas.getBoundingClientRect();

        // Set the canvas resolution to match display size with device pixel ratio for sharp rendering
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        // Scale the context to account for device pixel ratio
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
    }

    // 3D projection with rotation
    project3D(x, y, z, canvas) {
        // Use display size (not pixel-scaled size) for calculations
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        const centerX = width / 2;
        const centerY = height / 2;

        // Responsive scale - smaller on mobile, larger on desktop
        // Base scale on canvas size for consistency
        const baseScale = Math.min(width, height) / 10;
        const scale = baseScale *1.8; // Slight zoom out to ensure all points visible

        // Convert degrees to radians
        const radX = this.rotationX * Math.PI / 180;
        const radZ = this.rotationZ * Math.PI / 180;

        // Rotate around Z axis
        let x1 = x * Math.cos(radZ) - y * Math.sin(radZ);
        let y1 = x * Math.sin(radZ) + y * Math.cos(radZ);
        let z1 = z;

        // Rotate around X axis
        let y2 = y1 * Math.cos(radX) - z1 * Math.sin(radX);
        let z2 = y1 * Math.sin(radX) + z1 * Math.cos(radX);
        let x2 = x1;

        // Simple perspective projection
        const perspective = 5;
        const projScale = perspective / (perspective + z2 * 0.3);

        return {
            x: centerX + x2 * scale * projScale,
            y: centerY - y2 * scale * projScale,
            z: z2,
            scale: projScale
        };
    }

    render() {
        const canvas = document.getElementById('plot3d');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw axes
        this.drawAxes(ctx, canvas);

        // Prepare points with z-coordinate
        const points = this.transformedData.map(p => {
            const x = p.x;
            const y = p.y;
            const z = p.tz !== undefined ? p.tz : 0;
            const proj = this.project3D(x, y, z, canvas);
            return { ...p, proj, z3d: z };
        });

        // Sort by z for proper depth rendering (back to front)
        points.sort((a, b) => a.proj.z - b.proj.z);

        // Draw separating plane if separable
        if (this.isSeparable) {
            this.drawSeparatingPlane(ctx, canvas);
        }

        // Draw points
        for (const point of points) {
            // Responsive point size based on canvas size
            const baseSize = Math.min(width, height) / 50;
            const size = baseSize * point.proj.scale;

            ctx.beginPath();
            ctx.arc(point.proj.x, point.proj.y, size, 0, Math.PI * 2);

            // Use dark/light colors instead of red/green
            ctx.fillStyle = point.class === 0 ? this.darkColor : this.lightColor;
            ctx.fill();

            ctx.strokeStyle = '#f4e9d5';
            ctx.lineWidth = Math.max(1, baseSize / 4) * point.proj.scale;
            ctx.stroke();

            // Add subtle shadow for depth
            ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
            ctx.shadowBlur = 5;
            ctx.shadowOffsetX = 2;
            ctx.shadowOffsetY = 2;
        }

        // Reset shadow
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
    }

    drawAxes(ctx, canvas) {
        const axisLength = 2.5;
        const axes = [
            { start: [-axisLength, 0, 0], end: [axisLength, 0, 0], label: 'X', color: 'rgba(244, 233, 213, 0.5)' },
            { start: [0, -axisLength, 0], end: [0, axisLength, 0], label: 'Y', color: 'rgba(244, 233, 213, 0.5)' },
            { start: [0, 0, -axisLength], end: [0, 0, axisLength], label: 'Z', color: 'rgba(244, 233, 213, 0.7)' }
        ];

        ctx.lineWidth = 1;

        for (const axis of axes) {
            const start = this.project3D(...axis.start, canvas);
            const end = this.project3D(...axis.end, canvas);

            ctx.strokeStyle = axis.color;
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.stroke();

            // Label - responsive font size
            const rect = canvas.getBoundingClientRect();
            const fontSize = Math.max(10, Math.min(rect.width, rect.height) / 25);
            ctx.fillStyle = axis.color;
            ctx.font = `${fontSize}px Oswald, sans-serif`;
            ctx.fillText(axis.label, end.x + 5, end.y - 5);
        }

        // Draw grid on XY plane (z=0)
        ctx.strokeStyle = 'rgba(244, 233, 213, 0.1)';
        ctx.lineWidth = 0.5;

        for (let i = -2; i <= 2; i++) {
            const p1 = this.project3D(i, -2.5, 0, canvas);
            const p2 = this.project3D(i, 2.5, 0, canvas);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();

            const p3 = this.project3D(-2.5, i, 0, canvas);
            const p4 = this.project3D(2.5, i, 0, canvas);
            ctx.beginPath();
            ctx.moveTo(p3.x, p3.y);
            ctx.lineTo(p4.x, p4.y);
            ctx.stroke();
        }
    }

    drawSeparatingPlane(ctx, canvas) {
        // Draw a simple horizontal plane at z = 0 (the separating hyperplane)
        const planePoints = [
            this.project3D(-2, -2, 0, canvas),
            this.project3D(2, -2, 0, canvas),
            this.project3D(2, 2, 0, canvas),
            this.project3D(-2, 2, 0, canvas)
        ];

        ctx.fillStyle = this.planeColor;
        ctx.beginPath();
        ctx.moveTo(planePoints[0].x, planePoints[0].y);
        for (let i = 1; i < planePoints.length; i++) {
            ctx.lineTo(planePoints[i].x, planePoints[i].y);
        }
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = 'rgba(244, 233, 213, 0.6)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    applyTransform(userCode) {
        try {
            // Create the transform function
            const transformFunc = new Function('x', 'y', userCode + '\nreturn transform(x, y);');

            // Test the function first
            const testResult = transformFunc(1, 1);
            if (!Array.isArray(testResult) || testResult.length < 1) {
                throw new Error('Function must return an array with at least one value for Z');
            }

            // Apply transformation to all points
            this.transformedData = this.originalData.map(point => {
                const result = transformFunc(point.x, point.y);
                // If result has 1 element, use as Z. If 2+ elements, use first as new feature
                const tz = Array.isArray(result) ? result[0] : result;
                return { ...point, tz };
            });

            // Check for NaN or Infinity
            for (const point of this.transformedData) {
                if (!isFinite(point.tz)) {
                    throw new Error('Transformation produced invalid values (NaN or Infinity)');
                }
            }

            // Check if now separable (z > 0 for one class, z < 0 for other)
            this.isSeparable = this.checkLinearSeparability();

            // Mark level complete BEFORE rendering message so redirect logic works
            if (this.isSeparable) {
                this.handleLevelComplete();
            }

            this.render();
            this.renderStats();
            this.renderDataTable();
            this.renderMessage();

            document.getElementById('errorMessage').textContent = '';

        } catch (error) {
            document.getElementById('errorMessage').textContent = 'Error: ' + error.message;
            this.renderMessage('Error: ' + error.message, 'error');
        }
    }

    handleLevelComplete() {
        this.levelsCompleted[this.currentLevel] = true;
        this.saveProgress();
        this.renderLevelIndicator();

        // Check if all levels are complete
        const allComplete = this.levelsCompleted.every(c => c);

        if (allComplete) {
            // Unlock thoughts
            localStorage.setItem('thoughtsUnlocked', 'true');
        }
    }

    checkLinearSeparability() {
        // Check if all class 0 points have z < 0 and all class 1 points have z > 0 (or vice versa)
        const class0 = this.transformedData.filter(p => p.class === 0);
        const class1 = this.transformedData.filter(p => p.class === 1);

        const class0AllNegative = class0.every(p => p.tz < 0);
        const class1AllPositive = class1.every(p => p.tz > 0);

        const class0AllPositive = class0.every(p => p.tz > 0);
        const class1AllNegative = class1.every(p => p.tz < 0);

        return (class0AllNegative && class1AllPositive) || (class0AllPositive && class1AllNegative);
    }

    renderStats() {
        const sepElement = document.getElementById('isSeparable');
        if (sepElement) {
            sepElement.textContent = this.isSeparable ? 'Yes!' : 'No';
            sepElement.className = 'stat-value' + (this.isSeparable ? ' success' : '');
        }

        const darkCount = this.originalData.filter(p => p.class === 0).length;
        const lightCount = this.originalData.filter(p => p.class === 1).length;

        const darkElement = document.getElementById('darkCount');
        const lightElement = document.getElementById('lightCount');
        if (darkElement) darkElement.textContent = darkCount;
        if (lightElement) lightElement.textContent = lightCount;
    }

    renderDataTable() {
        const container = document.getElementById('dataTable');
        if (!container) return;

        let html = '<h3>Data Points</h3>';
        html += '<table class="data-table"><thead><tr>';
        html += '<th>X</th><th>Y</th><th>Z (transformed)</th><th>Class</th>';
        html += '</tr></thead><tbody>';

        for (const point of this.transformedData) {
            const rowClass = point.class === 0 ? 'class-dark' : 'class-light';
            const tz = point.tz !== undefined ? point.tz.toFixed(3) : '0.000';

            html += `<tr class="${rowClass}">`;
            html += `<td>${point.x.toFixed(3)}</td>`;
            html += `<td>${point.y.toFixed(3)}</td>`;
            html += `<td>${tz}</td>`;
            html += `<td>${point.class === 0 ? 'Dark' : 'Light'}</td>`;
            html += '</tr>';
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    renderMessage(customMessage, type) {
        const container = document.getElementById('gameMessage');
        if (!container) return;

        if (customMessage) {
            container.innerHTML = `<div class="game-message ${type || 'info'}">${customMessage}</div>`;
            return;
        }

        if (this.isSeparable) {
            const allComplete = this.levelsCompleted.every(c => c);

            if (allComplete) {
                container.innerHTML = `<div class="game-message success">All levels complete! Unlocking Thoughts...</div>`;
                setTimeout(() => {
                    window.location.href = 'series/Thoughts.html';
                }, 2500);
            } else if (this.currentLevel < this.levels.length - 1) {
                container.innerHTML = `<div class="game-message success">Level ${this.currentLevel + 1} Complete! <button class="game-btn next-level-btn" onclick="game.loadLevel(${this.currentLevel + 1})">Next Level</button></div>`;
            } else {
                // On last level but previous levels not done
                const incomplete = this.levelsCompleted.map((c, i) => !c ? i + 1 : null).filter(x => x !== null);
                container.innerHTML = `<div class="game-message info">Level complete! Complete level(s) ${incomplete.join(', ')} to unlock Thoughts.</div>`;
            }
        } else {
            container.innerHTML = `<div class="game-message info">Create a Z value that separates dark points (below Z=0) from light points (above Z=0). Drag to rotate the view.</div>`;
        }
    }

    showHint() {
        this.hintLevel++;
        const level = this.levels[this.currentLevel];

        let hintText = '';

        switch (this.hintLevel) {
            case 1:
                hintText = level.hint1;
                break;
            case 2:
                hintText = level.hint2;
                break;
            case 3:
            default:
                hintText = level.hint3;
                break;
        }

        const overlay = document.createElement('div');
        overlay.className = 'hint-overlay';
        overlay.innerHTML = `
            <div class="hint-box">
                <h3>Hint ${Math.min(this.hintLevel, 3)}/3</h3>
                <p>${hintText}</p>
                <button class="game-btn" onclick="this.parentElement.parentElement.remove()">Got it!</button>
            </div>
        `;
        document.body.appendChild(overlay);

        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
    }

    reset() {
        this.transformedData = this.originalData.map(p => ({ ...p, tz: 0 }));
        this.isSeparable = false;
        this.rotationX = -20;
        this.rotationZ = 45;

        const codeArea = document.getElementById('userCode');
        if (codeArea) {
            codeArea.value = '  // Your Z coordinate formula\n  // Example: return [x + y];\n  return [0];';
        }

        const errorMsg = document.getElementById('errorMessage');
        if (errorMsg) errorMsg.textContent = '';

        this.render();
        this.renderStats();
        this.renderDataTable();
        this.renderMessage();
    }

    // Reset all progress (for testing)
    resetAllProgress() {
        this.levelsCompleted = [false, false, false];
        this.currentLevel = 0;
        localStorage.removeItem('hyperplaneProgress');
        localStorage.removeItem('thoughtsUnlocked');
        this.loadLevel(0);
    }
}

// Initialize game
let game;
document.addEventListener('DOMContentLoaded', () => {
    game = new HyperplaneGame();
    game.init();
});

function applyTransform() {
    const userCode = document.getElementById('userCode').value;
    const fullCode = `function transform(x, y) {\n${userCode}\n}`;
    game.applyTransform(fullCode);
}

function resetGame() {
    game.reset();
}

function showHint() {
    game.showHint();
}
