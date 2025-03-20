// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = '#ffffff';
        navbar.style.boxShadow = 'none';
    }
});

// Chat button click handler
document.querySelector('.chat-btn').addEventListener('click', function() {
    alert('Chat feature coming soon!');
});

// Feature card hover effect
const featureCards = document.querySelectorAll('.feature-card');
featureCards.forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-5px)';
    });
    
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// Login button click handler
document.querySelector('.login-btn').addEventListener('click', function() {
    alert('Login/Register feature coming soon!');
});

// Search functionality
const searchInput = document.querySelector('.search-bar input');
searchInput.addEventListener('keyup', function(e) {
    if (e.key === 'Enter') {
        alert(`Searching for: ${this.value}`);
    }
});

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const searchInput = document.querySelector('.search-bar input');
    const chatBtn = document.querySelector('.chat-btn');
    const authBtn = document.querySelector('.auth-btn');
    const navLinks = document.querySelectorAll('.nav-links a');
    const navbar = document.querySelector('.navbar');

    // Navbar scroll effect
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = '0 2px 5px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = '#ffffff';
            navbar.style.boxShadow = 'none';
        }
    });

    // Handle search functionality
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const searchTerm = this.value.trim();
            if (searchTerm) {
                console.log('Searching for:', searchTerm);
            }
        }
    });

    // Handle chat button click
    chatBtn.addEventListener('click', function() {
        console.log('Opening chat with ArthaSathi');
    });

    // Handle auth button click
    authBtn.addEventListener('click', function() {
        console.log('Opening authentication modal');
    });

    // Handle navigation link clicks and smooth scrolling
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            // Add active class to clicked link
            this.classList.add('active');
            
            // Handle smooth scrolling for hash links
            if (this.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });

    // Feature card hover effect
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}); 