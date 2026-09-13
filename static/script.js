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
    const loading = document.getElementById('loading');

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
});
