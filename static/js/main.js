// Form handling and general interactions
document.addEventListener('DOMContentLoaded', () => {
    // Search functionality for students list
    const searchInput = document.getElementById('searchInput');
    if(searchInput) {
        searchInput.addEventListener('input', filterStudents);
    }

    // Form submission handling
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            
            try {
                const response = await fetch(form.action, {
                    method: form.method,
                    body: formData
                });
                
                if(response.redirected) {
                    window.location.href = response.url;
                }
            } catch (error) {
                console.error('Form submission error:', error);
            }
        });
    });
});

function filterStudents() {
    const input = this.value.toLowerCase();
    const rows = document.querySelectorAll('.students-table tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(input) ? '' : 'none';
    });
}