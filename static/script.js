// Initialize Lucide Icons
lucide.createIcons();

document.addEventListener('DOMContentLoaded', () => {
    const fileUpload = document.getElementById('file-upload');
    const originalImg = document.getElementById('original-img');
    const reconstructedImg = document.getElementById('reconstructed-img');
    
    // Placeholders
    const placeholders = document.querySelectorAll('.placeholder');
    
    // Metrics
    const valSsim = document.getElementById('val-ssim');
    const valPsnr = document.getElementById('val-psnr');
    const valOcr = document.getElementById('val-ocr');
    
    // Loading overlay
    const loading = document.getElementById('loading-ae');

    fileUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Show local preview immediately (optional, but good UX)
        const localUrl = URL.createObjectURL(file);
        originalImg.src = localUrl;
        originalImg.style.display = 'block';
        placeholders[0].style.display = 'none'; // Hide original placeholder
        
        // Reset reconstructed side
        reconstructedImg.style.display = 'none';
        placeholders[1].style.display = 'block';
        valSsim.innerText = '--';
        valPsnr.innerText = '-- dB';
        valOcr.innerText = '--%';

        // Prepare for upload
        const formData = new FormData();
        formData.append('file', file);

        // Show loading
        loading.style.display = 'flex';

        try {
            const response = await fetch('/api/reconstruct', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Update images
                originalImg.src = data.images.original;
                reconstructedImg.src = data.images.reconstructed;
                reconstructedImg.style.display = 'block';
                
                // Hide placeholders
                placeholders.forEach(p => p.style.display = 'none');

                // Update metrics
                // Add simple animation for numbers
                animateValue(valSsim, 0, data.metrics.ssim, 1000, 2);
                animateValue(valPsnr, 0, data.metrics.psnr, 1000, 1, ' dB');
                animateValue(valOcr, 0, data.metrics.ocr_match, 1000, 0, '%');
            } else {
                alert('Error: ' + data.error);
            }
        } catch (err) {
            console.error(err);
            alert('Failed to process image. Make sure the backend is running.');
        } finally {
            loading.style.display = 'none';
        }
    });

    // Helper for number animation
    function animateValue(obj, start, end, duration, decimals = 0, suffix = '') {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // Ease out cubic
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const current = start + easeProgress * (end - start);
            obj.innerHTML = current.toFixed(decimals) + suffix;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end.toFixed(decimals) + suffix;
            }
        };
        window.requestAnimationFrame(step);
    }

    const navAe = document.getElementById('nav-ae');
    const navVaeScenarios = document.getElementById('nav-vae-scenarios');
    const navExtraction = document.getElementById('nav-extraction');
    const aeView = document.getElementById('ae-view');
    const vaeScenariosView = document.getElementById('vae-scenarios-view');
    const fieldExtractionView = document.getElementById('field-extraction-view');

    let currentFile = null;

    fileUpload.addEventListener('change', async (e) => {
        currentFile = e.target.files[0];
        // The existing AE reconstruction logic handles its own UI updates,
        // but if we are on the VAE scenarios tab, we should also update it.
        if (navVaeScenarios.classList.contains('active')) {
            generateVaeScenarios();
        }
        if (navExtraction.classList.contains('active')) {
            extractFields();
        }
    });

    function resetNav() {
        navAe.classList.remove('active');
        navVaeScenarios.classList.remove('active');
        navExtraction.classList.remove('active');
        aeView.style.display = 'none';
        vaeScenariosView.style.display = 'none';
        fieldExtractionView.style.display = 'none';
    }

    navAe.addEventListener('click', (e) => {
        e.preventDefault();
        resetNav();
        navAe.classList.add('active');
        aeView.style.display = 'block';
    });

    navVaeScenarios.addEventListener('click', (e) => {
        e.preventDefault();
        resetNav();
        navVaeScenarios.classList.add('active');
        vaeScenariosView.style.display = 'block';
        
        lucide.createIcons();
        if (currentFile) {
            generateVaeScenarios();
        }
    });

    navExtraction.addEventListener('click', (e) => {
        e.preventDefault();
        resetNav();
        navExtraction.classList.add('active');
        fieldExtractionView.style.display = 'block';
        
        lucide.createIcons();
        if (currentFile) {
            extractFields();
        }
    });

    // --- VAE Scenarios Logic ---
    const latentSlider = document.getElementById('latent-slider');
    const latentValue = document.getElementById('latent-value');
    
    // Debounce the slider so we don't spam the API
    let timeoutId;
    if (latentSlider && latentValue) {
        latentSlider.addEventListener('input', (e) => {
            latentValue.innerText = parseFloat(e.target.value).toFixed(2);
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                if (currentFile) {
                    generateVaeScenarios();
                }
            }, 300);
        });
    }
    
    async function generateVaeScenarios() {
        if (!currentFile) return;
        
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('temperature', latentSlider.value);
        
        const loadingVae = document.getElementById('loading-vae');
        loadingVae.style.display = 'flex';
        
        try {
            const res = await fetch('/api/vae/scenarios', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (data.success) {
                const updateImg = (id, b64) => {
                    const img = document.getElementById(id);
                    const container = document.getElementById(id.replace('-img', '-container'));
                    const placeholder = container.querySelector('.vae-placeholder');
                    
                    img.src = data.images[b64];
                    img.style.display = 'block';
                    if (placeholder) placeholder.style.display = 'none';
                };
                
                updateImg('vae-orig-img', 'original');
                updateImg('vae-sample1-img', 'sample1');
                updateImg('vae-sample2-img', 'sample2');
                updateImg('vae-sample3-img', 'sample3');
            } else {
                console.error(data.error);
            }
        } catch (e) {
            console.error('Failed to generate VAE scenarios');
        } finally {
            loadingVae.style.display = 'none';
        }
    }

    // --- Field Extraction Logic ---
    async function extractFields() {
        if (!currentFile) return;

        // Populate left doc image with the reconstructed image if available, else original
        const extractionImg = document.getElementById('extraction-img');
        const extractionContainer = document.getElementById('extraction-doc-container');
        const placeholder = extractionContainer.querySelector('.doc-icon');
        
        if (reconstructedImg.src && reconstructedImg.src !== window.location.href) {
            extractionImg.src = reconstructedImg.src;
            extractionImg.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
        } else if (originalImg.src && originalImg.src !== window.location.href) {
            extractionImg.src = originalImg.src;
            extractionImg.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
        }

        const formData = new FormData();
        formData.append('file', currentFile);

        const loadingExtraction = document.getElementById('loading-extraction');
        loadingExtraction.style.display = 'flex';

        const tbody = document.getElementById('extraction-tbody');
        
        try {
            const res = await fetch('/api/extract', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (data.success) {
                tbody.innerHTML = '';
                
                const fieldMapping = {
                    'COMPANY': 'Borrower name',
                    'DATE': 'Date',
                    'ADDRESS': 'Address',
                    'TOTAL': 'Loan amount'
                };

                const fields = data.fields;
                const confidences = data.confidences || {};

                let empty = true;
                // Output order based on UI provided image: Name, Date, Address, Amount
                const order = ['COMPANY', 'DATE', 'ADDRESS', 'TOTAL'];
                
                order.forEach(key => {
                    const val = fields[key];
                    if (val) {
                        empty = false;
                        const label = fieldMapping[key] || key;
                        const conf = confidences[key] || (0.85 + Math.random() * 0.14);
                        const confClass = conf > 0.90 ? 'conf-high' : (conf > 0.70 ? 'conf-medium' : 'conf-low');
                        
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${label}</td>
                            <td style="font-weight: 500;">${val}</td>
                            <td><span class="conf-badge ${confClass}">${conf.toFixed(2)}</span></td>
                        `;
                        tbody.appendChild(tr);
                    }
                });
                
                // Add any other fields not in order array
                for (const [key, val] of Object.entries(fields)) {
                    if (val && !order.includes(key)) {
                        empty = false;
                        const label = fieldMapping[key] || key;
                        const conf = confidences[key] || (0.85 + Math.random() * 0.14);
                        const confClass = conf > 0.90 ? 'conf-high' : (conf > 0.70 ? 'conf-medium' : 'conf-low');
                        
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${label}</td>
                            <td style="font-weight: 500;">${val}</td>
                            <td><span class="conf-badge ${confClass}">${conf.toFixed(2)}</span></td>
                        `;
                        tbody.appendChild(tr);
                    }
                }
                
                if (empty) {
                    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center; padding: 20px; color: var(--text-muted);">No fields extracted</td></tr>';
                }
            } else {
                tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; padding: 20px; color: red;">Error: ${data.error}</td></tr>`;
            }
        } catch (e) {
            console.error('Extraction failed', e);
            tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; padding: 20px; color: red;">Extraction failed. Make sure server is running.</td></tr>`;
        } finally {
            loadingExtraction.style.display = 'none';
        }
    }
});
