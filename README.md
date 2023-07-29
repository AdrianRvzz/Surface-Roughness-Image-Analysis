<h1>Surface Roughness Image Analysis</h1>

<h2>Description</h2>
<p>Surface Roughness Image Analysis is a Python program developed for the analysis and processing of images related to surface roughness of microscopic specimens. It provides tools to calculate elements like waviness, roughness, and profile, enabling comprehensive information about the irregularity of the specimen's surface. This project was developed as part of a Ph.D. thesis in additive manufacturing.</p>

<h2>Preview</h2>

![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/c325d639-684f-4dcb-ae5b-9881e3d7bd5a)
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/f12b49a1-5242-48a7-b1fb-42791ea1da95)
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/79c2c469-9712-4c97-9f3e-6c451df34ca0)


<h2>Table of Contents</h2>
    <ol>
        <li><a href="#features">Features</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#how-to-use">How to Use</a></li>
        <li><a href="#requirements">Features</a></li>
        <li><a href="#author">Author</a></li>
    </ol>

 <h2 id="installation">Installation</h2>
    <p>To install the Surface Roughness Image Analysis program, follow these steps:</p>
    <ol>
        <li>Clone this repository to your local machine using git.</li>
        <li>Install the required dependencies using pip.</li>
        <li>Place the required images in the specified directory.</li>
        <li>Open the Python code and modify the <code>pathImageFile</code> variable to set the file path of the image you want
            to analyze.</li>
    </ol>
<h2 id="features">Features</h2>
<ul>
        <li>Binarization of images to obtain specimen edges.</li>
        <li>Polynomial fitting to calculate the initial profile.</li>
        <li>One-dimensional interpolation for centering the profile.</li>
        <li>Gaussian filtering to compute the waviness profile.</li>
        <li>Simple interface with adjustable parameters for sigma and cutoff of the Gaussian function.</li>
        <li>Visualization of four graphs showing the shape, profile, waviness, and roughness of the specimen.</li>
        <li>Visualization of average, cuadratic mean, max and minium peak and valley of profile, waviness and roughness.</li>    
</ul>

<h2 id="how-to-use">How to Use</h2>
<ol>
        <li>Obtain a photo or image corresponding to the piece you want to analyze 
         
<img src="https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/60498dd5-3c2e-4d3a-8ba4-ac36dc2b0a6c"> 







  
  </li>
        <li">Use 
<a href="https://imagej.nih.gov/ij/download.html">ImageJ</a>
       to convert the image to binary format.
          
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/ef8ecfad-ef39-402d-8d17-66b817a358f8)
        
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/c9c00c6f-4a7a-4163-8789-37fe94768ffe)
          
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/b82b2667-e07f-40c7-9c52-433a8cce3497)
  </li>
        <li>Find edges using the "Find Edges" function and invert with Ctrl + Shif + I.

![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/749e5585-2684-42ec-9933-1be41d954406)
        
  
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/f92f1f64-79cb-4cc0-81c1-dbeb3967dff0)

![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/3e3194a3-98cf-4012-b268-8f4bda12fe0f)

          
  </li>
      
  <li>Refine the image by removing any unwanted elements that should not be considered in the analysis.
  
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/628f814e-2219-454d-a027-fe1ef377a4da)
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/159f07df-8cad-4b22-8171-62b5af57b304)

          
  </li>
        <li>Once you are satisfied with the image, save it as a .tiff file in the repository directory.
          


  </li>
        <li>Provide the image's file path in the Python code by setting the variable <code>pathImageFile</code>.
          
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/f44d1a29-ae77-41ab-9145-7e17db6c65cf)
  </li>
        <li>Run the program.</li>
        <li>It will display an initial analysis with predefined sigma values.

![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/56b91cc5-ed73-477c-8548-b7eab8933a90)
      
  </li>
        <li>You can modify these values in the upper section.

![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/a137c654-0cb1-4803-a525-36ef054bbab1)

  </li>
        <li>Examples:
        
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/4a0bbfd9-d7c8-4269-abb4-dce47123fb14)
![image](https://github.com/AdrianRvzz/Surface-Roughness-Image-Analysis/assets/101829447/ca5ce0d9-76f2-40b5-bbeb-e4a772c99ea7)

</li>
    </ol>


  <h2 id="requirements">Requirements</h2>
    <p>The program requires the following software and dependencies:</p>
    <ul>
        <li>Python (version 3.10)</li>
        <li><a href="https://imagej.nih.gov/ij/download.html">ImageJ</a></li>
       
  </ul>



<h2>Autor</h2>
<p>Adrian Rivas E.</p>

