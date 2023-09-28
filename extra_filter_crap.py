galex = ['galex_FUV', 'galex_NUV']
hst_wfc3_uv  = ['wfc3_uvis_f275w', 'wfc3_uvis_f336w', 'wfc3_uvis_f475w','wfc3_uvis_f555w', 'wfc3_uvis_f606w', 'wfc3_uvis_f814w']
hst_wfc3_ir = ['wfc3_ir_f105w', 'wfc3_ir_f110w','wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']
spitzer_mips = ['spitzer_mips_24']
herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
jwst_nircam = ['jwst_f070w','jwst_f090w','jwst_f115w', 'jwst_f150w', 'jwst_f200w', 'jwst_f277w', 'jwst_f356w', 'jwst_f444w']
jwst_miri = ['jwst_f560w', 'jwst_f770w', 'jwst_f1000w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w', 'jwst_f2100w']


filternames = []#jwst_nircam
filternames = filternames+galex+hst_wfc3_uv+hst_wfc3_ir+spitzer_mips+herschel_pacs+herschel_spire+jwst_nircam+jwst_miri
#filternames = ['galex_FUV']
