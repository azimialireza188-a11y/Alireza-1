function template_picker_cb(src,~,fig)

    id = get(src,'Tag');        % 'lippedc', 'chs', etc
    if isempty(id), id = ''; end

    % store choice
    setappdata(fig,'TemplateID',id);

    % resume and close picker
    uiresume(fig);
    delete(fig);

    % launch template GUI
    template_section_gui(id,[]);

end