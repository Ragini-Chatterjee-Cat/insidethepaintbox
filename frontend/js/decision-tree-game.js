// Decision Tree Game - Build the shortest tree!

class DecisionTreeGame {
    constructor() {
        this.data = [];
        this.features = ['CocoaPercent', 'BeanType', 'Origin'];
        this.target = 'class'; // 1 if Rating >= 3.5, else 0
        this.tree = null;
        this.currentNode = null;
        this.treeDepth = 0;
        this.optimalDepth = 2; // Pre-calculated optimal depth
        this.splits = 0;
        this.gameOver = false;
        this.selectedFeature = null; // Track which feature is being configured
    }

    async init() {
        await this.loadData();
        this.prepareData();
        this.tree = this.createRootNode();
        this.currentNode = this.tree;
        this.render();
    }

    async loadData() {
        try {
            const response = await fetch('../data/chocolate.json');
            this.data = await response.json();
        } catch (error) {
            console.error('Error loading data:', error);
            // Fallback data
            this.data = [
                {id: 1, CocoaPercent: 70, Rating: 3.5, BeanType: "Criollo", Origin: "Venezuela"},
                {id: 2, CocoaPercent: 85, Rating: 2.5, BeanType: "Forastero", Origin: "Ghana"},
                {id: 3, CocoaPercent: 60, Rating: 3.75, BeanType: "Trinitario", Origin: "Ecuador"},
                {id: 4, CocoaPercent: 72, Rating: 4.0, BeanType: "Criollo", Origin: "Madagascar"},
                {id: 5, CocoaPercent: 80, Rating: 2.75, BeanType: "Forastero", Origin: "Ivory Coast"},
                {id: 6, CocoaPercent: 65, Rating: 3.25, BeanType: "Trinitario", Origin: "Peru"},
                {id: 7, CocoaPercent: 90, Rating: 2.0, BeanType: "Forastero", Origin: "Brazil"},
                {id: 8, CocoaPercent: 55, Rating: 3.5, BeanType: "Criollo", Origin: "Venezuela"},
                {id: 9, CocoaPercent: 78, Rating: 3.0, BeanType: "Trinitario", Origin: "Dominican Republic"},
                {id: 10, CocoaPercent: 68, Rating: 4.0, BeanType: "Criollo", Origin: "Madagascar"},
                {id: 11, CocoaPercent: 82, Rating: 2.25, BeanType: "Forastero", Origin: "Ghana"},
                {id: 12, CocoaPercent: 63, Rating: 3.75, BeanType: "Trinitario", Origin: "Ecuador"},
                {id: 13, CocoaPercent: 75, Rating: 3.25, BeanType: "Criollo", Origin: "Peru"},
                {id: 14, CocoaPercent: 88, Rating: 2.5, BeanType: "Forastero", Origin: "Ivory Coast"},
                {id: 15, CocoaPercent: 58, Rating: 3.5, BeanType: "Trinitario", Origin: "Venezuela"},
                {id: 16, CocoaPercent: 71, Rating: 3.75, BeanType: "Criollo", Origin: "Madagascar"}
            ];
        }
    }

    prepareData() {
        // Add class column: 1 if Rating >= 3.5, else 0
        this.data = this.data.map(row => ({
            ...row,
            class: row.Rating >= 3.5 ? 1 : 0
        }));
    }

    createRootNode() {
        return {
            id: 'root',
            samples: [...this.data],
            depth: 0,
            entropy: this.calculateEntropy(this.data),
            isPure: this.isPure(this.data),
            splitFeature: null,
            splitValue: null,
            children: [],
            parent: null
        };
    }

    calculateEntropy(samples) {
        if (samples.length === 0) return 0;
        const p1 = samples.filter(s => s.class === 1).length / samples.length;
        const p0 = 1 - p1;
        // Entropy: -Σ(p * log2(p))
        let entropy = 0;
        if (p0 > 0) entropy -= p0 * Math.log2(p0);
        if (p1 > 0) entropy -= p1 * Math.log2(p1);
        return entropy;
    }

    isPure(samples) {
        if (samples.length === 0) return true;
        const firstClass = samples[0].class;
        return samples.every(s => s.class === firstClass);
    }

    getClassCounts(samples) {
        const count0 = samples.filter(s => s.class === 0).length;
        const count1 = samples.filter(s => s.class === 1).length;
        return { low: count0, high: count1 };
    }

    findBestSplit(samples, feature) {
        if (feature === 'CocoaPercent') {
            return this.findBestNumericSplit(samples, feature);
        } else {
            return this.findBestCategoricalSplit(samples, feature);
        }
    }

    findBestNumericSplit(samples, feature) {
        const values = [...new Set(samples.map(s => s[feature]))].sort((a, b) => a - b);
        let bestEntropy = Infinity;
        let bestThreshold = null;

        for (let i = 0; i < values.length - 1; i++) {
            const threshold = (values[i] + values[i + 1]) / 2;
            const left = samples.filter(s => s[feature] <= threshold);
            const right = samples.filter(s => s[feature] > threshold);

            if (left.length === 0 || right.length === 0) continue;

            const weightedEntropy = (left.length / samples.length) * this.calculateEntropy(left) +
                                 (right.length / samples.length) * this.calculateEntropy(right);

            if (weightedEntropy < bestEntropy) {
                bestEntropy = weightedEntropy;
                bestThreshold = threshold;
            }
        }

        return { threshold: bestThreshold, entropy: bestEntropy, type: 'numeric' };
    }

    findBestCategoricalSplit(samples, feature) {
        const categories = [...new Set(samples.map(s => s[feature]))];

        // For simplicity, split into each category vs rest
        let bestEntropy = Infinity;
        let bestCategory = null;

        for (const cat of categories) {
            const left = samples.filter(s => s[feature] === cat);
            const right = samples.filter(s => s[feature] !== cat);

            if (left.length === 0 || right.length === 0) continue;

            const weightedEntropy = (left.length / samples.length) * this.calculateEntropy(left) +
                                 (right.length / samples.length) * this.calculateEntropy(right);

            if (weightedEntropy < bestEntropy) {
                bestEntropy = weightedEntropy;
                bestCategory = cat;
            }
        }

        return { category: bestCategory, entropy: bestEntropy, type: 'categorical' };
    }

    splitNode(node, feature) {
        if (node.isPure || node.samples.length <= 1) {
            return false;
        }

        const split = this.findBestSplit(node.samples, feature);
        if (!split.threshold && !split.category) return false;

        node.splitFeature = feature;

        let leftSamples, rightSamples, leftLabel, rightLabel;

        if (split.type === 'numeric') {
            node.splitValue = split.threshold;
            leftSamples = node.samples.filter(s => s[feature] <= split.threshold);
            rightSamples = node.samples.filter(s => s[feature] > split.threshold);
            leftLabel = `<= ${split.threshold.toFixed(1)}`;
            rightLabel = `> ${split.threshold.toFixed(1)}`;
        } else {
            node.splitValue = split.category;
            leftSamples = node.samples.filter(s => s[feature] === split.category);
            rightSamples = node.samples.filter(s => s[feature] !== split.category);
            leftLabel = `= ${split.category}`;
            rightLabel = `!= ${split.category}`;
        }

        const leftNode = {
            id: `${node.id}-L`,
            samples: leftSamples,
            depth: node.depth + 1,
            entropy: this.calculateEntropy(leftSamples),
            isPure: this.isPure(leftSamples),
            splitFeature: null,
            splitValue: null,
            children: [],
            parent: node,
            edgeLabel: leftLabel
        };

        const rightNode = {
            id: `${node.id}-R`,
            samples: rightSamples,
            depth: node.depth + 1,
            entropy: this.calculateEntropy(rightSamples),
            isPure: this.isPure(rightSamples),
            splitFeature: null,
            splitValue: null,
            children: [],
            parent: node,
            edgeLabel: rightLabel
        };

        node.children = [leftNode, rightNode];
        this.splits++;
        this.updateTreeDepth();
        this.checkGameOver();

        return true;
    }

    updateTreeDepth() {
        const getDepth = (node) => {
            if (node.children.length === 0) return node.depth;
            return Math.max(...node.children.map(getDepth));
        };
        this.treeDepth = getDepth(this.tree);
    }

    checkGameOver() {
        const allPure = (node) => {
            if (node.children.length === 0) return node.isPure;
            return node.children.every(allPure);
        };

        if (allPure(this.tree)) {
            this.gameOver = true;
        }
    }

    getImpureLeaves() {
        const leaves = [];
        const traverse = (node) => {
            if (node.children.length === 0 && !node.isPure) {
                leaves.push(node);
            }
            node.children.forEach(traverse);
        };
        traverse(this.tree);
        return leaves;
    }

    getAllLeaves() {
        const leaves = [];
        const traverse = (node) => {
            if (node.children.length === 0) {
                leaves.push(node);
            }
            node.children.forEach(traverse);
        };
        traverse(this.tree);
        return leaves;
    }

    reset() {
        this.tree = this.createRootNode();
        this.currentNode = this.tree;
        this.treeDepth = 0;
        this.splits = 0;
        this.gameOver = false;
        this.selectedFeature = null;
        this.render();
    }

    // Rendering methods
    render() {
        this.renderStats();
        this.renderTree();
        this.renderFeatureButtons();
        this.renderDataTable();
        this.renderMessage();
    }

    renderStats() {
        document.getElementById('treeSplits').textContent = this.splits;
        document.getElementById('treeDepth').textContent = this.treeDepth;
        document.getElementById('optimalDepth').textContent = this.optimalDepth;

        const impureLeaves = this.getImpureLeaves().length;
        document.getElementById('impureNodes').textContent = impureLeaves;
    }

    renderTree() {
        const container = document.getElementById('treeVisualization');
        const nodeWidth = 130;
        const nodeHeight = 60;
        const levelWidth = 180; // Horizontal spacing between levels

        // Calculate tree layout (horizontal: root on left, children to right)
        const nodes = [];
        const edges = [];

        // Count leaves to determine height
        const countLeaves = (node) => {
            if (node.children.length === 0) return 1;
            return node.children.reduce((sum, child) => sum + countLeaves(child), 0);
        };

        const totalLeaves = countLeaves(this.tree);
        const height = Math.max(400, totalLeaves * 80);

        const layoutTree = (node, x, yStart, yEnd) => {
            const y = (yStart + yEnd) / 2;
            nodes.push({ node, x, y });

            if (node.children.length === 2) {
                const [top, bottom] = node.children;
                const topLeaves = countLeaves(top);
                const bottomLeaves = countLeaves(bottom);
                const totalChildLeaves = topLeaves + bottomLeaves;

                const childX = x + levelWidth;
                const yMid = yStart + (yEnd - yStart) * (topLeaves / totalChildLeaves);

                edges.push({
                    from: { x: x + nodeWidth / 2, y },
                    to: { x: childX - nodeWidth / 2, y: (yStart + yMid) / 2 },
                    label: top.edgeLabel
                });
                edges.push({
                    from: { x: x + nodeWidth / 2, y },
                    to: { x: childX - nodeWidth / 2, y: (yMid + yEnd) / 2 },
                    label: bottom.edgeLabel
                });

                layoutTree(top, childX, yStart, yMid);
                layoutTree(bottom, childX, yMid, yEnd);
            }
        };

        layoutTree(this.tree, 50, 30, height - 30);

        // Calculate SVG width
        const maxX = Math.max(...nodes.map(n => n.x)) + nodeWidth + 50;

        // Build SVG
        let svg = `<svg class="tree-svg" width="${maxX}" height="${height}" viewBox="0 0 ${maxX} ${height}">`;

        // Draw edges
        for (const edge of edges) {
            svg += `<path class="tree-edge" d="M${edge.from.x},${edge.from.y} C${edge.from.x + 50},${edge.from.y} ${edge.to.x - 50},${edge.to.y} ${edge.to.x},${edge.to.y}"/>`;
            const midX = (edge.from.x + edge.to.x) / 2;
            const midY = (edge.from.y + edge.to.y) / 2;
            svg += `<text class="edge-label" x="${midX}" y="${midY - 5}">${edge.label || ''}</text>`;
        }

        // Draw nodes
        for (const { node, x, y } of nodes) {
            const isLeaf = node.children.length === 0;
            const boxClass = node.isPure ? 'pure' : 'impure';
            const isSelected = this.currentNode === node;
            const selectedClass = isSelected ? 'selected' : '';
            const counts = this.getClassCounts(node.samples);

            svg += `<g class="tree-node" data-node-id="${node.id}" onclick="game.selectNode('${node.id}')">`;
            svg += `<rect class="node-box ${boxClass} ${selectedClass}" x="${x - nodeWidth / 2}" y="${y - nodeHeight / 2}" width="${nodeWidth}" height="${nodeHeight}"/>`;

            if (node.splitFeature) {
                svg += `<text class="node-text" x="${x}" y="${y - 12}">${node.splitFeature}</text>`;
                svg += `<text class="node-text" x="${x}" y="${y + 5}" style="font-size:10px">${typeof node.splitValue === 'number' ? node.splitValue.toFixed(1) : node.splitValue}</text>`;
            } else {
                svg += `<text class="node-text" x="${x}" y="${y - 8}">n=${node.samples.length}</text>`;
                svg += `<text class="node-text" x="${x}" y="${y + 8}" style="font-size:10px">Low:${counts.low} High:${counts.high}</text>`;
            }

            svg += `<text class="node-entropy" x="${x}" y="${y + nodeHeight / 2 - 5}">Entropy: ${node.entropy.toFixed(3)}</text>`;
            svg += `</g>`;
        }

        svg += `</svg>`;
        container.innerHTML = svg;
    }

    renderFeatureButtons() {
        const container = document.getElementById('featureButtons');

        if (this.gameOver) {
            container.innerHTML = '<p>Game Complete! All nodes are pure.</p>';
            return;
        }

        if (!this.currentNode || this.currentNode.isPure) {
            container.innerHTML = '<p>Select an impure node (red) to split</p>';
            return;
        }

        let html = '';

        // If a feature is selected, show split options
        if (this.selectedFeature) {
            html += `<h3>Split on ${this.selectedFeature}:</h3>`;
            html += '<div class="split-options">';

            const samples = this.currentNode.samples;

            if (this.selectedFeature === 'CocoaPercent') {
                // Show numeric threshold options
                const values = [...new Set(samples.map(s => s.CocoaPercent))].sort((a, b) => a - b);
                html += '<p class="split-hint">Select threshold:</p>';
                html += '<div class="feature-buttons">';

                for (let i = 0; i < values.length - 1; i++) {
                    const threshold = Math.round((values[i] + values[i + 1]) / 2);
                    const leftCount = samples.filter(s => s.CocoaPercent <= threshold).length;
                    const rightCount = samples.length - leftCount;
                    html += `<button class="feature-btn split-btn" onclick="game.performSplitWithValue('CocoaPercent', ${threshold}, 'numeric')">
                        <= ${threshold}%<br><small>(${leftCount} | ${rightCount})</small>
                    </button>`;
                }
                html += '</div>';
            } else {
                // Show categorical options
                const categories = [...new Set(samples.map(s => s[this.selectedFeature]))];
                html += `<p class="split-hint">Select category:</p>`;
                html += '<div class="feature-buttons">';

                for (const cat of categories) {
                    const leftCount = samples.filter(s => s[this.selectedFeature] === cat).length;
                    const rightCount = samples.length - leftCount;
                    html += `<button class="feature-btn split-btn" onclick="game.performSplitWithValue('${this.selectedFeature}', '${cat}', 'categorical')">
                        ${cat}<br><small>(${leftCount} | ${rightCount})</small>
                    </button>`;
                }
                html += '</div>';
            }

            html += `<button class="game-btn" onclick="game.cancelFeatureSelection()" style="margin-top: 15px;">Back</button>`;
        } else {
            // Show feature selection
            html += '<h3>Choose a feature to split on:</h3><div class="feature-buttons">';

            for (const feature of this.features) {
                html += `<button class="feature-btn" onclick="game.selectFeature('${feature}')">${feature}</button>`;
            }

            html += '</div>';
        }

        container.innerHTML = html;
    }

    selectFeature(feature) {
        this.selectedFeature = feature;
        this.renderFeatureButtons();
    }

    cancelFeatureSelection() {
        this.selectedFeature = null;
        this.renderFeatureButtons();
    }

    performSplitWithValue(feature, value, type) {
        if (!this.currentNode || this.currentNode.isPure || this.currentNode.samples.length <= 1) {
            return;
        }

        const node = this.currentNode;
        node.splitFeature = feature;
        node.splitValue = value;

        let leftSamples, rightSamples, leftLabel, rightLabel;

        if (type === 'numeric') {
            leftSamples = node.samples.filter(s => s[feature] <= value);
            rightSamples = node.samples.filter(s => s[feature] > value);
            leftLabel = `<= ${value}`;
            rightLabel = `> ${value}`;
        } else {
            leftSamples = node.samples.filter(s => s[feature] === value);
            rightSamples = node.samples.filter(s => s[feature] !== value);
            leftLabel = `= ${value}`;
            rightLabel = `!= ${value}`;
        }

        if (leftSamples.length === 0 || rightSamples.length === 0) {
            return;
        }

        const leftNode = {
            id: `${node.id}-L`,
            samples: leftSamples,
            depth: node.depth + 1,
            entropy: this.calculateEntropy(leftSamples),
            isPure: this.isPure(leftSamples),
            splitFeature: null,
            splitValue: null,
            children: [],
            parent: node,
            edgeLabel: leftLabel
        };

        const rightNode = {
            id: `${node.id}-R`,
            samples: rightSamples,
            depth: node.depth + 1,
            entropy: this.calculateEntropy(rightSamples),
            isPure: this.isPure(rightSamples),
            splitFeature: null,
            splitValue: null,
            children: [],
            parent: node,
            edgeLabel: rightLabel
        };

        node.children = [leftNode, rightNode];
        this.splits++;
        this.updateTreeDepth();
        this.checkGameOver();

        // Reset selection and select next impure node
        this.selectedFeature = null;
        const impureLeaves = this.getImpureLeaves();
        this.currentNode = impureLeaves.length > 0 ? impureLeaves[0] : null;
        this.render();
    }

    renderDataTable() {
        const container = document.getElementById('dataTable');
        const samples = this.currentNode ? this.currentNode.samples : this.data;

        let html = `<h3>Current Node Data (${samples.length} samples)</h3>`;
        html += '<table class="data-table"><thead><tr>';
        html += '<th>ID</th><th>Cocoa %</th><th>Bean Type</th><th>Origin</th><th>Rating</th><th>Class</th>';
        html += '</tr></thead><tbody>';

        for (const row of samples) {
            const rowClass = row.class === 1 ? 'class-high' : 'class-low';
            html += `<tr class="${rowClass}">`;
            html += `<td>${row.id}</td>`;
            html += `<td>${row.CocoaPercent}%</td>`;
            html += `<td>${row.BeanType}</td>`;
            html += `<td>${row.Origin}</td>`;
            html += `<td>${row.Rating}</td>`;
            html += `<td>${row.class === 1 ? 'High' : 'Low'}</td>`;
            html += '</tr>';
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    renderMessage() {
        const container = document.getElementById('gameMessage');

        if (this.gameOver) {
            let message = `Congratulations! You built a tree with depth ${this.treeDepth} using ${this.splits} splits.`;
            if (this.treeDepth <= this.optimalDepth) {
                message += ' You found an optimal solution! Unlocking Diary Entries...';
                // Save unlock status to localStorage
                localStorage.setItem('diaryEntriesUnlocked', 'true');
                // Redirect to diary entries after success
                setTimeout(() => {
                    window.location.href = 'series/diary-entries.html';
                }, 2000);
            } else {
                message += ` The optimal depth is ${this.optimalDepth}. Try again to find a shorter tree!`;
            }
            container.innerHTML = `<div class="game-message success">${message}</div>`;
        } else if (this.currentNode && !this.currentNode.isPure) {
            const counts = this.getClassCounts(this.currentNode.samples);
            container.innerHTML = `<div class="game-message info">Selected node has ${this.currentNode.samples.length} samples (Low: ${counts.low}, High: ${counts.high}). Entropy: ${this.currentNode.entropy.toFixed(3)}</div>`;
        } else {
            container.innerHTML = '<div class="game-message info">Click on a red (impure) node to select it, then choose a feature to split.</div>';
        }
    }

    selectNode(nodeId) {
        const findNode = (node) => {
            if (node.id === nodeId) return node;
            for (const child of node.children) {
                const found = findNode(child);
                if (found) return found;
            }
            return null;
        };

        const node = findNode(this.tree);
        if (node && !node.isPure && node.children.length === 0) {
            this.currentNode = node;
            this.render();
        }
    }

    performSplit(feature) {
        if (this.currentNode && this.splitNode(this.currentNode, feature)) {
            // Select first impure leaf
            const impureLeaves = this.getImpureLeaves();
            this.currentNode = impureLeaves.length > 0 ? impureLeaves[0] : null;
            this.render();
        }
    }
}

// Initialize game
let game;
document.addEventListener('DOMContentLoaded', () => {
    game = new DecisionTreeGame();
    game.init();
});

function resetGame() {
    game.reset();
}
