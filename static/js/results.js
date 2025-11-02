document.addEventListener('DOMContentLoaded', function() {
    // Add class based on detection result
    const summaryContainer = document.getElementById('results-summary-container');
    
    // This information needs to be passed from the server-side
    // We'll assume there's a hidden input or data attribute with this info
    const isCancerDetected = summaryContainer.getAttribute('data-cancer-detected') === 'true' || 
                            document.querySelector('.fa-exclamation-triangle') !== null;
    
    if (summaryContainer) {
        if (isCancerDetected) {
            summaryContainer.classList.add('cancer-detected');
        } else {
            summaryContainer.classList.add('no-cancer');
        }
    }
    
    // Animate confidence meter
    const meterFill = document.querySelector('.meter-fill');
    if (meterFill) {
        const width = meterFill.getAttribute('data-value') + '%';
        meterFill.style.width = '0';
        
        setTimeout(() => {
            meterFill.style.transition = 'width 1.5s ease-out';
            meterFill.style.width = width;
        }, 500);
    }
    
    // Animate feature bars
    const featureFills = document.querySelectorAll('.feature-fill');
    if (featureFills.length > 0) {
        featureFills.forEach(fill => {
            const width = fill.getAttribute('data-value') + '%';
            fill.style.width = '0';
            
            setTimeout(() => {
                fill.style.transition = 'width 1.5s ease-out';
                fill.style.width = width;
            }, 800);
        });
    }
    
    // Print functionality
    const printBtn = document.getElementById('print-report');
    if (printBtn) {
        printBtn.addEventListener('click', function() {
            window.print();
        });
    }
    
    // Download report functionality
    const downloadBtn = document.getElementById('download-report');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            generatePDF();
        });
    }
    
    function generatePDF() {
        alert('Report download functionality would be implemented here in a production environment.');
        // In a real application, you would generate a PDF using a library like jsPDF or html2pdf
        
        // Example implementation with html2pdf (would require the library to be included)
        /*
        const element = document.querySelector('.results-container');
        const opt = {
            margin:       1,
            filename:     'breast-cancer-analysis-report.pdf',
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2 },
            jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
        };
        
        // New Promise-based usage:
        html2pdf().set(opt).from(element).save();
        */
    }
    
    // Check if GSAP is available
    if (typeof gsap !== 'undefined') {
        // Animate elements with GSAP
        gsap.from(".results-header", {opacity: 0, y: -30, duration: 0.8});
        gsap.from(".results-summary", {opacity: 0, y: -20, duration: 0.8, delay: 0.3});
        gsap.from(".result-images", {opacity: 0, x: -30, duration: 0.8, delay: 0.5});
        gsap.from(".metrics-container", {opacity: 0, x: 30, duration: 0.8, delay: 0.7});
        gsap.from(".recommendation-section", {opacity: 0, y: 30, duration: 0.8, delay: 0.9});
        gsap.from(".technical-details", {opacity: 0, y: 20, duration: 0.8, delay: 1.1});
        gsap.from(".action-buttons", {opacity: 0, y: 20, duration: 0.8, delay: 1.3});
        gsap.from(".related-resources", {opacity: 0, y: 30, duration: 0.8, delay: 1.5});
    } else {
        console.warn("GSAP library not loaded. Animations will not run.");
    }
});
