listing = dir('./a_train_images/*.png');
for i = 1:length(listing)
    str = strcat('./a_train_images/',listing(i).name);
    str2 = strcat('./a_train_images_b/',listing(i).name);
    img = imread(str);
    tabimg = img(1:end,1:end,1);
    binimg = 255*(1-sauvola(tabimg, [15,15], 0.65));
    imwrite(binimg,str2)
end