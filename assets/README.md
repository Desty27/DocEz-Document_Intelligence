# Assets Directory

## Company Logo

To add your company logo to the GESIL RAG System interface:

1. **Add your logo file**: Place your company logo image in this directory with the name `logo.png`
   
   Supported formats: PNG (recommended), JPG, JPEG
   
   Recommended dimensions: 
   - Width: 200-400 pixels
   - Height: 80-150 pixels
   - Aspect ratio: 2:1 to 4:1 (landscape orientation works best)

2. **File naming**: 
   - Default expected name: `logo.png`
   - If you want to use a different name or format, update the `logo_path` variable in `app.py` line 183

3. **Example logo placement**:
   ```
   assets/
   ├── logo.png          (your company logo)
   └── README.md          (this file)
   ```

4. **No logo**: If no logo file is found, the interface will display just the title without a logo.

## Notes

- The logo will be automatically resized to fit the header while maintaining aspect ratio
- Logo is displayed above the "GESIL RAG System" title
- For best results, use a transparent background PNG with your company logo
