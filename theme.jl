set_theme!()
CairoMakie.activate!(; type="png")
update_theme!(;
    backgroundcolor=:transparent,
    fonts=(;
        regular="STIX Two Text",
        bold="STIX Two Text SemiBold",
        italic="STIX Two Text Italic",
        bold_italic="STIX Two Text SemiBold Italic",
    ),
    fontsize=12,
    Figure=(;
        backgroundcolor=:transparent
    ),
    Axis=(;
        backgroundcolor=:transparent,
        xlabelpadding= 8,
        ylabelpadding= 8,
        xlabelsize = 14,
        ylabelsize = 14,
        xticklabelsize = 13,
        yticklabelsize = 13,
        xgridstyle = :dot,
        ygridstyle = :dot,
        titlealign = :left,
        titlesize = 17,
        titlegap = 10,
    ),
    CairoMakie=(; px_per_unit=6),
)