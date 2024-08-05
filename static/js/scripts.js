const pivotIterations = {
    'vertices_plot_800.png': { defaultBasis: '74,799', sibBasis: '150,000' },
    'vertices_plot_801.png': { defaultBasis: '35,067', sibBasis: '15,853' },
    'vertices_plot_802.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: 'Time Limit Exceeded' },
    'vertices_plot_803.png': { defaultBasis: '14,072', sibBasis: '15,585' },
    'vertices_plot_804.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: 'Time Limit Exceeded' },
    'vertices_plot_805.png': { defaultBasis: '13,826', sibBasis: '12,228' },
    'vertices_plot_806.png': { defaultBasis: '38,602', sibBasis: '12,784' },
    'vertices_plot_807.png': { defaultBasis: '12,521', sibBasis: '17,874' },
    'vertices_plot_808.png': { defaultBasis: '37,936', sibBasis: '12,644' },
    'vertices_plot_809.png': { defaultBasis: '39,878', sibBasis: '16,187' },
    'vertices_plot_810.png': { defaultBasis: '16,543', sibBasis: '14,661' },
    'vertices_plot_811.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: 'Time Limit Exceeded' },
    'vertices_plot_812.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: 'Time Limit Exceeded' },
    'vertices_plot_813.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: 'Time Limit Exceeded' },
    'vertices_plot_814.png': { defaultBasis: '12,163', sibBasis: '9,396' },
    'vertices_plot_815.png': { defaultBasis: '45,627', sibBasis: '13,231' },
    'vertices_plot_816.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: 'Time Limit Exceeded' },
    'vertices_plot_817.png': { defaultBasis: '28,654', sibBasis: '12,142' },
    'vertices_plot_818.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: '113,307' },
    'vertices_plot_819.png': { defaultBasis: 'Time Limit Exceeded', sibBasis: 'Time Limit Exceeded' }
};

let currentZoomLevel = 'images_zoom_out'; // Initial zoom level
let currentVertexVisibility = 'hidden_vertices'; // Initial vertices visibility

function showImage(imageName) {
    const img = document.getElementById('visualization');
    img.src = `../static/images/${currentZoomLevel}/${currentVertexVisibility}/${imageName}`;
    updateText();
    updateDescription(imageName);
    
    // Hide README and show visualization
    document.getElementById('readme-content').style.display = 'none';
    document.getElementById('visualization-section').style.display = 'block';
}

function toggleZoom() {
    const zoomLevels = ['images_zoom_out', 'images_zoom_mid', 'images_zoom_in'];
    const currentZoomIndex = zoomLevels.indexOf(currentZoomLevel);
    currentZoomLevel = zoomLevels[(currentZoomIndex + 1) % zoomLevels.length];
    const img = document.getElementById('visualization');
    const currentImageSrc = img.src;
    const imageName = currentImageSrc.substring(currentImageSrc.lastIndexOf('/') + 1);
    showImage(imageName);
}

function toggleVertexVisibility() {
    const vertexVisibilities = ['unhidden_vertices', 'hidden_vertices'];
    const currentVisibilityIndex = vertexVisibilities.indexOf(currentVertexVisibility);
    currentVertexVisibility = vertexVisibilities[(currentVisibilityIndex + 1) % vertexVisibilities.length];
    const img = document.getElementById('visualization');
    const currentImageSrc = img.src;
    const imageName = currentImageSrc.substring(currentImageSrc.lastIndexOf('/') + 1);
    showImage(imageName);
}

function updateText() {
    const zoomTextElement = document.getElementById('zoom-level-text');
    const vertexVisibilityTextElement = document.getElementById('vertex-visibility-text');
    const zoomNames = {
        'images_zoom_out': 'Zoom Out',
        'images_zoom_mid': 'Zoom Mid',
        'images_zoom_in': 'Zoom In'
    };
    const vertexVisibilityNames = {
        'unhidden_vertices': 'Unhidden',
        'hidden_vertices': 'Hidden'
    };
    zoomTextElement.textContent = `Current Zoom Level: ${zoomNames[currentZoomLevel]}`;
    vertexVisibilityTextElement.textContent = `Vertices Visibility: ${vertexVisibilityNames[currentVertexVisibility]}`;
}

function updateDescription(imageName) {
    const defaultBasisElement = document.getElementById('default-basis');
    const sibBasisElement = document.getElementById('sib-basis');
    const pivotData = pivotIterations[imageName];
    defaultBasisElement.textContent = `Pivots to optimal with Default Basis: ${pivotData.defaultBasis}`;
    sibBasisElement.textContent = `Pivots to optimal with Smart Initial Basis: ${pivotData.sibBasis}`;
}

function showREADME() {
    var readmeContent = document.getElementById('readme-content');
    var visualizationSection = document.getElementById('visualization-section');
    
    // Show README and hide visualization
    readmeContent.style.display = 'block';
    visualizationSection.style.display = 'none';
}