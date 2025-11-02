// Main JavaScript file for the Breast Cancer Diagnosis Application

document.addEventListener('DOMContentLoaded', function() {
    // Add animated background elements
    createAnimatedBackground();
    
    // Initialize any page-specific functionality
    initializePageFunctions();
    
    // Add scroll animations
    initScrollAnimations();
});

function createAnimatedBackground() {
    const backgroundContainer = document.createElement('ul');
    backgroundContainer.className = 'animated-bg';
    
    // Create 10 animated background elements
    for (let i = 0; i < 10; i++) {
        const li = document.createElement('li');
        backgroundContainer.appendChild(li);
    }
    
    document.body.appendChild(backgroundContainer);
}

function initializePageFunctions() {
    // Check which page we're on and initialize appropriate functionality
    const currentPath = window.location.pathname;
    
    // Home page specific functionality
    if (currentPath === '/' || currentPath.includes('index')) {
        initHomePageFunctions();
    }
    
    // Analyze page specific functionality
    if (currentPath.includes('analyze')) {
        initAnalyzePageFunctions();
    }
    
    // Results page specific functionality
    if (currentPath.includes('results')) {
        initResultsPageFunctions();
    }
    
    // Contact page specific functionality
    if (currentPath.includes('contact')) {
        initContactPageFunctions();
    }
}

function initHomePageFunctions() {
    // Animated counters
    const counters = document.querySelectorAll('.counter');
    if (counters.length > 0) {
        counters.forEach(counter => {
            const target = +counter.dataset.target;
            const increment = target / 100;
            
            const updateCounter = () => {
                const count = +counter.innerText;
                if (count < target) {
                    counter.innerText = Math.ceil(count + increment);
                    setTimeout(updateCounter, 20);
                } else {
                    counter.innerText = target;
                }
            };
            
            // Start counter when element is in viewport
            const observer = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting) {
                    updateCounter();
                    observer.disconnect();
                }
            }, { threshold: 0.5 });
            
            observer.observe(counter);
        });
    }
    
    // Testimonial slider
    const testimonialSlider = document.querySelector('.testimonial-slider');
    if (testimonialSlider) {
        const slides = document.querySelectorAll('.testimonial');
        const prevBtn = document.querySelector('.prev-btn');
        const nextBtn = document.querySelector('.next-btn');
        
        let currentSlide = 0;
        
        // Position slides initially
        slides.forEach((slide, i) => {
            slide.style.transform = `translateX(${100 * (i - currentSlide)}%)`;
        });
        
        function showSlide(index) {
            slides.forEach((slide, i) => {
                slide.style.transform = `translateX(${100 * (i - index)}%)`;
            });
        }
        
        if (prevBtn && nextBtn) {
            prevBtn.addEventListener('click', () => {
                currentSlide = (currentSlide > 0) ? currentSlide - 1 : slides.length - 1;
                showSlide(currentSlide);
            });
            
            nextBtn.addEventListener('click', () => {
                currentSlide = (currentSlide < slides.length - 1) ? currentSlide + 1 : 0;
                showSlide(currentSlide);
            });
            
            // Auto slide
            setInterval(() => {
                currentSlide = (currentSlide < slides.length - 1) ? currentSlide + 1 : 0;
                showSlide(currentSlide);
            }, 5000);
        }
    }
}

function initAnalyzePageFunctions() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const previewImage = document.getElementById('preview-image');
    const fileName = document.getElementById('file-name');
    const removeFile = document.getElementById('remove-file');
    const uploadForm = document.getElementById('upload-form');
    const loadingOverlay = document.getElementById('loading-overlay');
    const progress = document.getElementById('progress');
    
    if (dropArea && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('highlight');
        }
        
        // Handle dropped files
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }
        
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });
        
        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    displayPreview(file);
                } else {
                    alert('Please upload an image file.');
                }
            }
        }
        
        function displayPreview(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                fileName.textContent = file.name;
                dropArea.style.display = 'none';
                filePreview.style.display = 'flex';
            }
            reader.readAsDataURL(file);
        }
        
        if (removeFile) {
            removeFile.addEventListener('click', function() {
                fileInput.value = '';
                previewImage.src = '#';
                fileName.textContent = '';
                dropArea.style.display = 'flex';
                filePreview.style.display = 'none';
            });
        }
        
        if (uploadForm && loadingOverlay && progress) {
            uploadForm.addEventListener('submit', function(e) {
                if (fileInput.files.length > 0) {
                    loadingOverlay.style.display = 'flex';
                    simulateProgress();
                }
            });
            
            function simulateProgress() {
                let width = 0;
                const interval = setInterval(function() {
                    if (width >= 90) {
                        clearInterval(interval);
                    } else {
                        width += Math.random() * 10;
                        progress.style.width = width + '%';
                    }
                }, 500);
            }
        }
    }
}

function initResultsPageFunctions() {
    // Add animation to confidence meter
    const meterFill = document.querySelector('.meter-fill');
    if (meterFill) {
        const width = meterFill.style.width;
        meterFill.style.width = '0';
        
        setTimeout(() => {
            meterFill.style.transition = 'width 1.5s ease-out';
            meterFill.style.width = width;
        }, 500);
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
    }
}

function initContactPageFunctions() {
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Simple form validation
            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const message = document.getElementById('message').value;
            
            if (!name || !email || !message) {
                alert('Please fill in all fields');
                return;
            }
            
            // Email validation
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                alert('Please enter a valid email address');
                return;
            }
            
            // Show success message (in a real app, you would send the form data to a server)
            const formContainer = document.querySelector('.contact-form');
            formContainer.innerHTML = `
                <div class="success-message">
                    <i class="fas fa-check-circle" style="font-size: 3rem; color: var(--secondary-color); margin-bottom: 1rem;"></i>
                    <h3>Message Sent Successfully!</h3>
                    <p>Thank you for contacting us, ${name}. We will get back to you shortly.</p>
                </div>
            `;
        });
    }
}

function initScrollAnimations() {
    // Add fade-in animations to elements as they come into view
    const animatedElements = document.querySelectorAll('.card, .tech-card, .feature-cards, .about-content, .metrics-container');
    
    if (animatedElements.length > 0 && 'IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        animatedElements.forEach(el => {
            el.classList.add('animate-on-scroll');
            observer.observe(el);
        });
    }
}

// Utility function to check if an element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}
