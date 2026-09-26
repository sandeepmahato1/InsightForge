document.addEventListener('DOMContentLoaded', () => {
    
    const datasetSelect = document.getElementById('datasetSelect');
    const fileUpload = document.getElementById('fileUpload');
    const researchPrompt = document.getElementById('researchPrompt');
    const runBtn = document.getElementById('runBtn');
    const traceTimeline = document.getElementById('traceTimeline');
    const liveIndicator = document.getElementById('liveIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const hypothesesGrid = document.getElementById('hypothesesGrid');
    const mlBenchmarkTableEl = document.getElementById('mlBenchmarkTable');
    const mlBenchmarkTable = mlBenchmarkTableEl ? mlBenchmarkTableEl.querySelector('tbody') : null;
    const executiveSummaryContent = document.getElementById('executiveSummaryContent');
    const exportBtn = document.getElementById('exportBtn');
    
    // Section 3 Chat Elements
    const chatSection = document.getElementById('chatSection');
    const chatStatusPill = document.getElementById('chatStatusPill');
    const chatPlaceholder = document.getElementById('chatPlaceholder');
    const chatMessages = document.getElementById('chatMessages');
    const chatBody = document.getElementById('chatBody');
    const chatInput = document.getElementById('chatInput');
    const sendChatBtn = document.getElementById('sendChatBtn');

    let shapChartInstance = null;
    let currentInvestigationId = null;
    let currentChatHistory = [];
    let currentResults = null;

    function getApiUrl(path) {
        if (window.location.protocol === 'file:' || !window.location.port) {
            return 'http://127.0.0.1:8000' + path;
        }
        return path;
    }

    // Fetch available datasets on load
    fetchDatasets();

    // Quick Prompt Chips
    document.querySelectorAll('.quick-chips .chip').forEach(chip => {
        chip.addEventListener('click', () => {
            researchPrompt.value = chip.getAttribute('data-prompt');
        });
    });

    // File Upload Handler
    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(getApiUrl('/api/upload'), {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (res.ok) {
                alert(`Dataset '${file.name}' uploaded successfully!`);
                await fetchDatasets();
                datasetSelect.value = file.name;
            } else {
                alert(`Upload failed: ${data.detail || 'Error'}`);
            }
        } catch (err) {
            console.error('Upload Error:', err);
            alert('File upload failed.');
        }
    });

    async function fetchDatasets() {
        try {
            const res = await fetch(getApiUrl('/api/datasets'));
            const data = await res.json();
            if (data.datasets && data.datasets.length > 0) {
                datasetSelect.innerHTML = '';
                data.datasets.forEach(ds => {
                    const opt = document.createElement('option');
                    opt.value = ds;
                    opt.textContent = ds;
                    datasetSelect.appendChild(opt);
                });
            }
        } catch (err) {
            console.warn('Could not fetch datasets list.', err);
        }
    }

    // Run Initial Autonomous Investigation
    runBtn.addEventListener('click', async () => {
        const selectedDataset = datasetSelect.value || 'customer_churn.csv';
        const query = researchPrompt.value.trim() || 'Analyze this dataset and identify key patterns, insights, and predictive drivers.';

        // UI Reset
        runBtn.disabled = true;
        runBtn.innerHTML = `<span class="pulse"></span> Investigating...`;
        liveIndicator.style.display = 'flex';
        traceTimeline.innerHTML = '';
        resultsContainer.style.display = 'none';

        // Update Chat UI to Progress State
        chatStatusPill.className = 'chat-status-pill';
        chatStatusPill.innerHTML = `<span class="pulse-dot"></span> Investigation in Progress...`;
        chatInput.disabled = true;
        sendChatBtn.disabled = true;
        chatInput.placeholder = 'Investigation in progress...';

        try {
            const res = await fetch(getApiUrl('/api/investigate'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_name: selectedDataset,
                    research_question: query
                })
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `Server returned status ${res.status}`);
            }
            
            const data = await res.json();
            currentResults = data;
            currentInvestigationId = data.investigation_id;
            currentChatHistory = [];

            // Animate Execution Trace Step-by-Step
            await animateTraceLogs(data.agent_trace);

            // Enable Conversational Interface
            enableChatInterface(data);

            // Render Technical Evidence below
            renderTechnicalEvidence(data);

        } catch (err) {
            console.error('Investigation error:', err);
            traceTimeline.innerHTML = `<div class="trace-item" style="border-color: var(--accent-rose)">
                <div class="trace-agent" style="color: var(--accent-rose)">System Error</div>
                <div class="trace-details">Error: ${err.message || err}. Make sure server is running on http://127.0.0.1:8000.</div>
            </div>`;
        } finally {
            runBtn.disabled = false;
            runBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Start Autonomous Investigation`;
            liveIndicator.style.display = 'none';
        }
    });

    async function animateTraceLogs(logs) {
        for (const log of logs) {
            const item = document.createElement('div');
            item.className = 'trace-item';
            
            if (log.agent.includes('Root')) item.style.borderLeftColor = 'var(--accent-glow)';
            if (log.agent.includes('Data')) item.style.borderLeftColor = 'var(--accent-cyan)';
            if (log.agent.includes('Classifier') || log.agent.includes('Task')) item.style.borderLeftColor = '#38bdf8';
            if (log.agent.includes('Analysis')) item.style.borderLeftColor = 'var(--accent-amber)';
            if (log.agent.includes('Hypothesis')) item.style.borderLeftColor = '#f59e0b';
            if (log.agent.includes('Model Selection')) item.style.borderLeftColor = '#c084fc';
            if (log.agent.includes('Experiment')) item.style.borderLeftColor = 'var(--accent-emerald)';
            if (log.agent.includes('Explainability')) item.style.borderLeftColor = '#818cf8';
            if (log.agent.includes('Answer')) item.style.borderLeftColor = '#fbbf24';
            if (log.agent.includes('Report')) item.style.borderLeftColor = 'var(--accent-rose)';

            item.innerHTML = `
                <div class="trace-header">
                    <span class="trace-agent">${log.agent}</span>
                    <span class="trace-time">${log.timestamp}</span>
                </div>
                <div class="trace-step">${log.step}</div>
                <div class="trace-details">${log.details}</div>
            `;
            traceTimeline.appendChild(item);
            traceTimeline.scrollTop = traceTimeline.scrollHeight;

            await new Promise(r => setTimeout(r, 400));
        }
    }

    function enableChatInterface(data) {
        chatStatusPill.className = 'chat-status-pill status-active';
        chatStatusPill.innerHTML = `<span class="pulse-dot"></span> Investigation Complete — Ask Questions`;
        
        chatPlaceholder.style.display = 'none';
        chatMessages.style.display = 'flex';
        chatMessages.innerHTML = '';
        
        chatInput.disabled = false;
        sendChatBtn.disabled = false;
        chatInput.placeholder = 'Ask anything about your dataset or investigation findings...';

        // Add Initial AI Summary Message
        const initialAns = data.ai_answer ? data.ai_answer.direct_answer : data.executive_summary;
        appendChatMessage('assistant', initialAns);
    }

    // Send Chat Message Handler & Dynamic Textarea Resizing
    const chatInputBox = document.getElementById('chatInputBox');
    sendChatBtn.addEventListener('click', sendChatMessage);
    
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    chatInput.addEventListener('input', () => {
        chatInput.style.height = '24px';
        const newH = Math.min(chatInput.scrollHeight, 160);
        chatInput.style.height = newH + 'px';
        if (chatInputBox) {
            chatInputBox.style.alignItems = newH > 32 ? 'flex-end' : 'center';
            chatInputBox.style.borderRadius = newH > 32 ? '16px' : '24px';
        }
    });

    async function sendChatMessage() {
        const question = chatInput.value.trim();
        if (!question || !currentInvestigationId) return;

        // Render User Bubble
        appendChatMessage('user', question);
        chatInput.value = '';
        chatInput.style.height = '24px';
        if (chatInputBox) {
            chatInputBox.style.alignItems = 'center';
            chatInputBox.style.borderRadius = '24px';
        }
        chatInput.disabled = true;
        sendChatBtn.disabled = true;
        chatInput.placeholder = 'AI Data Scientist is thinking...';

        try {
            const res = await fetch(getApiUrl('/api/chat'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    investigation_id: currentInvestigationId,
                    question: question,
                    history: currentChatHistory
                })
            });

            if (!res.ok) {
                throw new Error(`Status ${res.status}`);
            }

            const data = await res.json();
            const aiAnswer = data.answer || "I don't have enough context to answer that.";
            
            // Update History
            currentChatHistory = data.chat_history || [];

            // Render AI Bubble with optional Chart Spec
            appendChatMessage('assistant', aiAnswer, data.chart_spec);

        } catch (err) {
            console.error('Chat error:', err);
            appendChatMessage('assistant', 'AI answer generation is temporarily unavailable. Please try again.');
        } finally {
            chatInput.disabled = false;
            sendChatBtn.disabled = false;
            chatInput.placeholder = 'Ask a follow-up question...';
            chatInput.focus();
        }
    }

    function appendChatMessage(role, content, chartSpec = null) {
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'}`;
        bubble.innerHTML = formatMarkdown(content);

        if (role === 'assistant' && chartSpec && typeof chartSpec === 'object') {
            const chartWrapper = document.createElement('div');
            chartWrapper.className = 'chat-chart-wrapper';
            renderChatChart(chartWrapper, chartSpec);
            bubble.appendChild(chartWrapper);
        }

        chatMessages.appendChild(bubble);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function renderChatChart(container, spec) {
        if (!spec || !spec.chart_type || !spec.data) return;

        const allowedTypes = ['bar', 'line', 'scatter', 'histogram', 'pie', 'heatmap'];
        const type = String(spec.chart_type).toLowerCase();
        if (!allowedTypes.includes(type)) return;

        const titleEl = document.createElement('div');
        titleEl.className = 'chat-chart-title';
        titleEl.textContent = spec.title || '';
        container.appendChild(titleEl);

        if (type === 'heatmap') {
            renderHeatmapMatrix(container, spec);
            return;
        }

        const canvasBox = document.createElement('div');
        canvasBox.className = 'chat-canvas-box';
        const canvas = document.createElement('canvas');
        canvasBox.appendChild(canvas);
        container.appendChild(canvasBox);

        const ctx = canvas.getContext('2d');
        const data = spec.data || [];
        const labels = data.map(d => String(d.x || ''));
        const values = data.map(d => (typeof d.y === 'number' ? d.y : 0));

        let chartType = type === 'histogram' ? 'bar' : type;
        let chartData = {
            labels: labels,
            datasets: [{
                label: spec.y || 'Value',
                data: values,
                backgroundColor: type === 'pie' ? 
                    ['#6366f1', '#38bdf8', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6'] : 
                    'rgba(99, 102, 241, 0.7)',
                borderColor: '#6366f1',
                borderWidth: 1,
                borderRadius: type === 'bar' ? 4 : 0
            }]
        };

        if (type === 'scatter') {
            chartData = {
                datasets: [{
                    label: `${spec.x} vs ${spec.y}`,
                    data: data.map(d => ({ x: Number(d.x), y: Number(d.y) })),
                    backgroundColor: 'rgba(56, 189, 248, 0.8)',
                    borderColor: '#38bdf8',
                    pointRadius: 5
                }]
            };
        }

        new Chart(ctx, {
            type: chartType,
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: type === 'pie' }
                },
                scales: type === 'pie' ? {} : {
                    x: {
                        ticks: { color: '#94a3b8', font: { size: 11 } },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        ticks: { color: '#94a3b8', font: { size: 11 } },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                }
            }
        });
    }

    function renderHeatmapMatrix(container, spec) {
        const data = spec.data || [];
        const rows = (spec.options && spec.options.rows) ? spec.options.rows : Array.from(new Set(data.map(d => d.row)));
        const cols = (spec.options && spec.options.columns) ? spec.options.columns : Array.from(new Set(data.map(d => d.col)));

        const valMap = {};
        let minVal = Infinity, maxVal = -Infinity;
        data.forEach(d => {
            const k = `${d.row}___${d.col}`;
            valMap[k] = d.value;
            if (typeof d.value === 'number') {
                if (d.value < minVal) minVal = d.value;
                if (d.value > maxVal) maxVal = d.value;
            }
        });
        if (minVal === Infinity) minVal = 0;
        if (maxVal === -Infinity) maxVal = 1;

        const table = document.createElement('table');
        table.className = 'heatmap-matrix-table';

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.appendChild(document.createElement('th'));
        cols.forEach(c => {
            const th = document.createElement('th');
            th.textContent = c;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        rows.forEach(r => {
            const tr = document.createElement('tr');
            const th = document.createElement('th');
            th.textContent = r;
            tr.appendChild(th);

            cols.forEach(c => {
                const td = document.createElement('td');
                const val = valMap[`${r}___${c}`];
                if (val !== undefined && val !== null && typeof val === 'number') {
                    td.textContent = Number.isInteger(val) ? val : val.toFixed(2);
                    const norm = maxVal > minVal ? (val - minVal) / (maxVal - minVal) : 0.5;
                    const alpha = Math.max(0.2, Math.min(0.85, norm));
                    td.style.backgroundColor = `rgba(99, 102, 241, ${alpha})`;
                    td.style.color = alpha > 0.5 ? '#ffffff' : '#e2e8f0';
                } else {
                    td.textContent = '-';
                    td.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
                    td.style.color = '#64748b';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        const tableWrapper = document.createElement('div');
        tableWrapper.className = 'heatmap-table-wrapper';
        tableWrapper.appendChild(table);
        container.appendChild(tableWrapper);
    }

    function renderTechnicalEvidence(data) {
        resultsContainer.style.display = 'block';

        // 1. Model Selection Strategy
        const modelSelectionSection = document.querySelector('.model-selection-panel');
        const modelSelectionBody = document.getElementById('modelSelectionBody');
        const ms = data.model_selection || {};
        
        if (ms.selected_candidates && ms.selected_candidates.length > 0) {
            modelSelectionSection.style.display = 'block';
            let candsPillsHtml = ms.selected_candidates.map(c => `<span class="candidate-pill">${c}</span>`).join('');
            let bulletsHtml = ms.selection_bullets ? ms.selection_bullets.map(b => `<li>${b}</li>`).join('') : `<li>${ms.selection_reasoning || 'Default selection strategy'}</li>`;
            
            modelSelectionBody.innerHTML = `
                <div class="model-meta-row">
                    <span class="meta-tag"><strong>Problem Category:</strong> ${data.problem_type || 'Classification'}</span>
                    <span class="meta-tag"><strong>Target Column:</strong> <code>${data.target_column || 'Target'}</code></span>
                    <span class="meta-tag meta-tag-metric"><strong>Primary Metric Strategy:</strong> ${ms.primary_metric || 'F1 Score'}</span>
                </div>
                <div class="model-selection-content">
                    <div class="candidate-list-wrapper">
                        <h4>Selected Candidate Library (${ms.selected_candidates.length} Algorithms):</h4>
                        <div class="candidate-pills">${candsPillsHtml}</div>
                    </div>
                    <div class="selection-reasoning-wrapper">
                        <h4>Architectural Selection Rationale:</h4>
                        <ul>${bulletsHtml}</ul>
                        <p class="metric-reason-note">📌 <strong>Evaluation Metric Rationale:</strong> ${ms.primary_metric_reason || 'Evaluated via cross-validation.'}</p>
                    </div>
                </div>
            `;
        } else {
            modelSelectionSection.style.display = 'none';
        }

        // 2. Hypotheses Grid
        hypothesesGrid.innerHTML = '';
        if (data.hypotheses && data.hypotheses.length > 0) {
            data.hypotheses.forEach(h => {
                const card = document.createElement('div');
                card.className = 'hypothesis-card';
                card.innerHTML = `
                    <div class="hypo-top">
                        <span class="hypo-id">${h.id}</span>
                        <span class="hypo-score">${h.confidence_score}% Confidence</span>
                    </div>
                    <h4 class="hypo-title">${h.title}</h4>
                    <p class="hypo-statement">${h.statement}</p>
                    <div class="hypo-evidence">📊 <strong>Evidence:</strong> ${h.supporting_evidence}</div>
                `;
                hypothesesGrid.appendChild(card);
            });
            hypothesesGrid.parentElement.style.display = 'block';
        } else {
            hypothesesGrid.parentElement.style.display = 'none';
        }

        // 3. ML Benchmark Table
        const mlPanel = document.getElementById('mlBenchmarkTable').closest('section');
        const tableElement = document.getElementById('mlBenchmarkTable');
        const theadElement = tableElement.querySelector('thead');
        
        if (data.ml_benchmarks && data.ml_benchmarks.length > 0) {
            mlPanel.style.display = 'block';
            const sampleBm = data.ml_benchmarks[0];
            const m1Name = sampleBm.metric_1_name || "Accuracy";
            const m2Name = sampleBm.metric_2_name || "Precision";
            const m3Name = sampleBm.metric_3_name || "Recall";
            const m4Name = sampleBm.metric_4_name || "F1 Score";
            const m5Name = sampleBm.metric_5_name || "ROC-AUC";

            theadElement.innerHTML = `
                <tr>
                    <th>Candidate Algorithm (${data.problem_type || 'Classification'})</th>
                    <th>Validation Status</th>
                    <th>${m1Name}</th>
                    <th>${m2Name}</th>
                    <th>${m3Name}</th>
                    <th>${m4Name}</th>
                    <th>${m5Name}</th>
                </tr>
            `;

            mlBenchmarkTable.innerHTML = '';
            data.ml_benchmarks.forEach(bm => {
                const tr = document.createElement('tr');
                if (bm.model_name === data.best_model_name) tr.className = 'winning-row';
                
                const val1 = bm.metric_1 !== undefined ? bm.metric_1 : bm.accuracy;
                const val2 = bm.metric_2 !== undefined ? bm.metric_2 : bm.precision;
                const val3 = bm.metric_3 !== undefined ? bm.metric_3 : bm.recall;
                const val4 = bm.metric_4 !== undefined ? bm.metric_4 : bm.f1_score;
                const val5 = bm.metric_5 !== undefined ? bm.metric_5 : bm.roc_auc;
                const statusStr = bm.status || 'Validated';

                tr.innerHTML = `
                    <td><strong>${bm.model_name}</strong> ${bm.model_name === data.best_model_name ? '🏆' : ''}</td>
                    <td><span class="status-pill">${statusStr}</span></td>
                    <td>${val1}</td>
                    <td>${val2}</td>
                    <td>${val3}</td>
                    <td><strong>${val4}</strong></td>
                    <td>${val5}</td>
                `;
                mlBenchmarkTable.appendChild(tr);
            });
        } else {
            mlPanel.style.display = 'none';
        }

        // 4. SHAP Chart Rendering
        const shapPanel = document.getElementById('shapChart').closest('section');
        if (data.feature_importance && data.feature_importance.length > 0) {
            shapPanel.style.display = 'block';
            renderShapChart(data.feature_importance);
        } else {
            shapPanel.style.display = 'none';
        }

        // 5. Executive Summary Markdown
        executiveSummaryContent.innerHTML = formatMarkdown(data.executive_summary);
    }

    function renderShapChart(features) {
        const ctx = document.getElementById('shapChart').getContext('2d');
        if (shapChartInstance) shapChartInstance.destroy();

        const labels = features.map(f => f.feature);
        const values = features.map(f => f.importance);

        shapChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'SHAP / Feature Attribution',
                    data: values,
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f1f5f9', font: { family: 'Fira Code' } }
                    }
                }
            }
        });
    }

    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatMarkdown(text) {
        if (!text) return '';
        const safeText = escapeHtml(text);
        return safeText
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '<br>')
            .replace(/\n/g, '<br>');
    }

    // Export handler
    exportBtn.addEventListener('click', () => {
        if (!currentResults) return;
        const blob = new Blob([JSON.stringify(currentResults, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `InsightForge_Investigation_${Date.now()}.json`;
        a.click();
    });

});
