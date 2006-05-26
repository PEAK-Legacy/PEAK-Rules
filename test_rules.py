def additional_tests():
    import doctest
    return doctest.DocFileSuite(
        'framework.txt', package='peak.rules',
        optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE,
    )

